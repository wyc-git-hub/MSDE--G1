import os
import time
import datetime
import json
import argparse
import torch
from src.unet import UNet
# 导入网络模型结构
from src.msde_net import MSDENet
# 导入训练和验证的工具函数（计算loss、更新参数、评估dice等）
from train_utils.train_and_eval import train_one_epoch, evaluate, create_lr_scheduler

# 导入我们刚刚专门为 IDRiD 多分类病灶定制的数据集类
from my_dataset import IDRiDDataset
# 导入数据预处理模块
import transforms as T
from train_utils.train_and_eval import AutoAdaptiveLoss


class SegmentationPresetTrain:
    """
    训练集的数据预处理与增强流水线
    """

    def __init__(self, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        trans = []
        # 注意：这里我们移除了 RandomResize 和 RandomCrop
        # 因为在 my_dataset.py 的 __getitem__ 中，我们已经强制将原图和标签 Resize 到了 800x800

        # 1. 随机水平翻转 (概率 50%)，增加数据多样性
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        # 2. 随机垂直翻转 (概率 50%)
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))

        # 3. 将 PIL Image 或 Numpy 转换为 PyTorch 的 Tensor 格式
        # 4. 标准化处理 (Normalize)，使用 ImageNet 的均值和标准差，有助于模型更快收敛
        trans.extend([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        # 使用 Compose 将所有步骤串联起来
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    """
    验证/测试集的数据预处理流水线
    (不需要做翻转等数据增强，只要转成 Tensor 并归一化即可)
    """

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    根据当前是训练模式还是验证模式，返回对应的预处理流水线
    """
    if train:
        return SegmentationPresetTrain(mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std)


def create_model(num_classes):
    """
    实例化 MSDENet 模型
    :param num_classes: 输出通道数 (病灶种类 + 1个背景)
    """
    model = UNet(in_channels=3, num_classes=num_classes, base_c=32)
    return model


def main(args):
    # 根据电脑配置，自动选择使用 GPU(cuda) 还是 CPU 训练
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size

    # 【核心修改点】：总类别数计算
    # args.num_classes 在 config.json 中设为了 4（代表 IDRiD 的 4 种病灶）
    # 网络最终需要输出的通道数 = 4 种病灶 + 1 个健康背景 = 5
    num_classes = args.num_classes + 1

    # 图像标准化的均值和标准差
    mean = (0.654, 0.317, 0.091)
    std = (0.113, 0.084, 0.038)

    # 自动生成一个带有当前时间戳的 txt 文件名，用于记录训练日志
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # 实例化训练集和验证集 (使用我们刚重写的 IDRiDDataset)
    train_dataset = IDRiDDataset(args.data_path,
                                 train=True,
                                 transforms=get_transform(train=True, mean=mean, std=std))

    val_dataset = IDRiDDataset(args.data_path,
                               train=False,
                               transforms=get_transform(train=False, mean=mean, std=std))

    # 自动计算合适的多进程数据加载 worker 数量 (最高不超过8)
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    # 构造 DataLoader，负责将 dataset 中的单张图片打包成批次 (Batch) 喂给 GPU
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,  # 训练集必须打乱顺序
                                               pin_memory=True,  # 锁页内存，加速数据传到 GPU
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,  # 验证时通常一张一张测
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    # 创建模型并搬运到对应设备(GPU/CPU)
    model = create_model(num_classes=num_classes)
    model.to(device)

    # ==================== 【新增核心修改 1：实例化自适应 Loss】 ====================
    loss_fn = AutoAdaptiveLoss(num_classes=num_classes).to(device)

    # ==================== 【新增核心修改 2：把模型和 Loss 的参数打包给优化器】 ====================
    params_to_optimize = [
        {"params": [p for p in model.parameters() if p.requires_grad]},
        {"params": loss_fn.parameters(), "weight_decay": 0.0}  # Loss内部的自动权重调节参数不需要正则化惩罚
    ]

    # 定义 SGD 优化器，负责在反向传播时更新网络权重
    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    # 如果开启了 AMP (自动混合精度)，则初始化 GradScaler，用于防止 FP16 精度下的梯度下溢出
    # ==================== 【新增核心修改 3：修复 FutureWarning】 ====================
    scaler = torch.amp.GradScaler('cuda') if args.amp else None

    # 创建学习率调度器：支持 Warmup 热身，以及 Poly 策略的动态衰减
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    # 如果配置了 resume 路径，则加载之前的断点权重继续训练 (断点续传)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    best_dice = 0.
    best_mpr = 0.0
    start_time = time.time()

    # ---------------- 开始正式训练循环 ----------------
    for epoch in range(args.start_epoch, args.epochs):
        # 1. 训练一个 Epoch (遍历一遍整个训练集)
        # ==================== 【新增核心修改 4：传入 loss_fn】 ====================
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes,
                                        lr_scheduler=lr_scheduler, loss_fn=loss_fn, print_freq=args.print_freq,
                                        scaler=scaler)

        # 接收新增的 mauc 和 mpr
        confmat, dice, mauc, mpr = evaluate(model, val_loader, device=device, num_classes=num_classes)

        val_info = str(confmat)
        print(val_info)
        print(f"Dice coefficient: {dice:.4f}")
        print(f"Mean ROC-AUC:     {mauc:.4f}")
        print(f"Mean PR-AUC:      {mpr:.4f}\n")

        with open(results_file, "a") as f:
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n" \
                         f"Dice coefficient: {dice:.4f}\n" \
                         f"Mean ROC-AUC: {mauc:.4f}\n" \
                         f"Mean PR-AUC: {mpr:.4f}\n"
            f.write(train_info + val_info + "\n\n")

        # 4. 判断当前模型是不是“历史最佳”
        if args.save_best is True:
            # 这里对比的是 mpr (Mean PR-AUC)
            if best_mpr < mpr:
                best_mpr = mpr
                print(f"--> [New Best Model] Mean PR-AUC updated to: {best_mpr:.4f}")
            else:
                continue  # 如果不是最好的，那就跳过保存，直接进入下一轮

        # 5. 打包需要保存的断点信息 (模型权重、优化器状态、当前轮次等)
        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        # 6. 保存到硬盘
        if args.save_best is True:
            torch.save(save_file, "save_weights/best_model.pth")
        else:
            torch.save(save_file, "save_weights/model_{}.pth".format(epoch))

    # 计算总耗时
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args(config_file="config.json"):
    """
    读取并解析外部的 config.json 配置文件，替代原本繁杂的命令行传参。
    """
    # 1. 定义一份保底的默认配置，防止 json 文件中漏写某些参数导致程序崩溃
    default_config = {
        "data_path": "./",  # 数据集根目录
        "num_classes": 4,  # 病灶的种类数 (不包含背景)
        "device": "cuda",  # 训练设备 (cuda 或 cpu)
        "batch_size": 4,  # 批处理大小 (显存不够请调小)
        "epochs": 200,  # 训练总轮数
        "lr": 0.01,  # 初始学习率
        "momentum": 0.9,  # 优化器动量
        "weight_decay": 0.0001,  # L2 正则化惩罚系数
        "print_freq": 1,  # 多少个 batch 打印一次日志
        "resume": "",  # 继续训练的权重路径
        "start_epoch": 0,  # 继续训练的起始 epoch
        "save_best": True,  # 是否仅保存 Dice 最高的模型
        "amp": False  # 是否开启自动混合精度训练 (加速且省显存)
    }

    # 2. 尝试读取 config.json
    if os.path.exists(config_file):
        print(f"=> 成功找到配置文件 '{config_file}'，正在加载...")
        with open(config_file, 'r', encoding='utf-8') as f:
            user_config = json.load(f)
            # 使用 json 中提取的配置，覆盖掉 default_config 中的同名项
            default_config.update(user_config)
    else:
        print(f"=> 警告: 根目录下未找到 '{config_file}'，将使用内部默认配置运行！")

    # 3. 将字典转化成 argparse.Namespace 对象
    # 这一步是为了完美兼容原代码中 args.batch_size 这种调用方式
    args = argparse.Namespace(**default_config)

    # 打印最终使用的超参数，方便排错
    print("-" * 50)
    print("Training Configuration (loaded from json):")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("-" * 50)

    return args


if __name__ == '__main__':
    # 1. 解析配置文件
    args = parse_args()

    # 2. 如果不存在保存权重的文件夹，自动创建一个
    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    # 3. 启动主干训练逻辑
    main(args)