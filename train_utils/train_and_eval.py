import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import train_utils.distributed_utils as utils
from torchmetrics.classification import MulticlassAUROC, MulticlassAveragePrecision


# ===================================================================== #
# [2026 SOTA] 自适应多目标损失函数模块
# 融合 GDL (解决极度体积不平衡) 与 不确定性自学习 (解决难易样本不平衡)
# ===================================================================== #

class AutoAdaptiveLoss(nn.Module):
    def __init__(self, num_classes=5, ignore_index=255, epsilon=1e-6):
        super(AutoAdaptiveLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.epsilon = epsilon

        # 可学习参数：各类别的 log方差（网络会自动优化这5个值，相当于自动调权重）
        self.log_vars = nn.Parameter(torch.zeros(num_classes))

        # 基础 CE Loss (不进行 reduction，因为我们要自己算加权)
        self.ce_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)

    def forward(self, inputs, targets):
        # 兼容深度学习模型可能输出的 dict 格式 (主输出 out，辅助输出 aux)
        if isinstance(inputs, dict):
            loss_out = self._compute_loss(inputs['out'], targets)
            if 'aux' in inputs:
                loss_aux = self._compute_loss(inputs['aux'], targets)
                return loss_out + 0.5 * loss_aux
            return loss_out
        else:
            return self._compute_loss(inputs, targets)

    def _compute_loss(self, logits, targets):
        # 1. 忽略 255 的无效填充区域
        valid_mask = (targets != self.ignore_index)

        # 2. 基础 Cross Entropy Loss
        base_ce = self.ce_loss(logits, targets)

        adaptive_ce = 0.0
        # 对每一个类别应用不确定性自学习加权
        for c in range(self.num_classes):
            mask_c = ((targets == c) & valid_mask).float()
            # 避免除以 0 的情况
            loss_c = (base_ce * mask_c).sum() / (mask_c.sum() + 1e-6)

            # 自学习核心公式：(1 / e^var) * loss + var
            precision = torch.exp(-self.log_vars[c])
            adaptive_ce += precision * loss_c + self.log_vars[c]

        # 3. 广义 Dice Loss (GDL) - 动态体积感知
        probs = F.softmax(logits, dim=1)

        # 为了顺利进行 one-hot 编码，先临时把 255 替换为 0
        targets_clean = targets.clone()
        targets_clean[targets == self.ignore_index] = 0
        targets_one_hot = F.one_hot(targets_clean, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        # 把 255 的区域真正 mask 掉
        valid_mask_ext = valid_mask.unsqueeze(1).float()
        probs = probs * valid_mask_ext
        targets_one_hot = targets_one_hot * valid_mask_ext

        # 动态计算每个类别的权重 (面积的倒数)
        volumes = torch.sum(targets_one_hot, dim=(0, 2, 3))
        volumes = torch.clamp(volumes, min=1.0)
        weights = 1.0 / (volumes ** 2 + self.epsilon)

        inter = torch.sum(probs * targets_one_hot, dim=(0, 2, 3)) * weights
        union = torch.sum(probs + targets_one_hot, dim=(0, 2, 3)) * weights

        gdl = 1.0 - (2.0 * inter.sum() + self.epsilon) / (union.sum() + self.epsilon)

        return adaptive_ce + gdl


# ===================================================================== #
# 训练与评估流程
# ===================================================================== #

def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    dice = utils.DiceCoefficient(num_classes=num_classes, ignore_index=255)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    auc_metric = MulticlassAUROC(num_classes=num_classes, average=None, ignore_index=255).to(device)
    pr_metric = MulticlassAveragePrecision(num_classes=num_classes, average=None, ignore_index=255).to(device)

    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 1, header):
            image, target = image.to(device), target.to(device)
            output = model(image)

            if isinstance(output, dict):
                output = output['out']

            confmat.update(target.flatten(), output.argmax(1).flatten())
            dice.update(output, target)

            auc_metric.update(output, target)
            pr_metric.update(output, target)

        confmat.reduce_from_all_processes()
        dice.reduce_from_all_processes()

    auc_scores = auc_metric.compute().cpu().numpy()
    pr_scores = pr_metric.compute().cpu().numpy()

    auc_metric.reset()
    pr_metric.reset()

    lesion_auc = [x if not np.isnan(x) else np.nan for x in auc_scores[1:]]
    lesion_pr = [x if not np.isnan(x) else np.nan for x in pr_scores[1:]]

    print(f"\n[Validation Metrics per Lesion]")
    print(f"Classes:      ['MA', 'HE', 'EX', 'SE']")
    print(f"ROC-AUC:      {[round(x, 4) if not np.isnan(x) else 'N/A' for x in lesion_auc]}")
    print(f"PR-AUC:       {[round(x, 4) if not np.isnan(x) else 'N/A' for x in lesion_pr]}")

    mauc = np.nanmean(lesion_auc)
    mpr = np.nanmean(lesion_pr)

    if np.isnan(mauc): mauc = 0.0
    if np.isnan(mpr): mpr = 0.0

    return confmat, dice.value.item(), mauc, mpr


# 【重构注意】：加入了 loss_fn 入参，删除了手动权重的逻辑
def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes,
                    lr_scheduler, loss_fn, print_freq=10, scaler=None):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            # 直接调用由外部传入的智能损失函数
            loss = loss_fn(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


def create_lr_scheduler(optimizer, num_step: int, epochs: int, warmup=True, warmup_epochs=1, warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)