import os
from PIL import Image
import numpy as np


def main():
    img_channels = 3

    # 【修改这里】：请将路径指向你解压后的 IDRiD 数据集根目录
    # 例如，和 config.json 里的 data_path 保持一致
    dataset_root = "./data/IDRiD/A. Segmentation"

    # 自动拼接官方的奇葩文件夹名称
    img_dir = os.path.join(dataset_root, "1. Original Images", "a. Training Set")

    assert os.path.exists(img_dir), f"image dir: '{img_dir}' does not exist. 请检查 dataset_root 路径配置是否正确！"

    # IDRiD 的原图是 .jpg 格式
    img_name_list = [i for i in os.listdir(img_dir) if i.endswith(".jpg")]

    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)

    print(f"成功找到路径，开始处理 {len(img_name_list)} 张图片...")

    for img_name in img_name_list:
        img_path = os.path.join(img_dir, img_name)

        # 1. 读取原图
        img_pil = Image.open(img_path).convert('RGB')
        img_np = np.array(img_pil)

        # 2. 动态生成 ROI Mask (不再需要外部的 Mask 文件)
        # 提取红色通道 (通道索引为0)
        red_channel = img_np[:, :, 0]
        # 设定阈值 10，大于 10 的认为是眼底有效区域 (值为 True)
        valid_mask = (red_channel > 10)

        # 3. 归一化图像到 [0, 1] 范围
        img_normalized = img_np / 255.0

        # 4. 极致过滤：只截取眼底有效区域内部的像素！
        # 这一步把四周黑边的像素全扔了，valid_pixels 的 shape 会变成 [有效像素总数, 3]
        valid_pixels = img_normalized[valid_mask]

        # 5. 计算这张图有效区域的均值和方差，并累加
        cumulative_mean += valid_pixels.mean(axis=0)
        cumulative_std += valid_pixels.std(axis=0)

    # 6. 求平均
    mean = cumulative_mean / len(img_name_list)
    std = cumulative_std / len(img_name_list)

    print("\n--- 计算完成 ---")
    print(f"原始输出 mean: {mean}")
    print(f"原始输出 std: {std}")

    print("\n请复制以下代码，直接替换 train.py 中的对应参数:")
    print("-" * 50)
    print(f"mean = ({mean[0]:.3f}, {mean[1]:.3f}, {mean[2]:.3f})")
    print(f"std = ({std[0]:.3f}, {std[1]:.3f}, {std[2]:.3f})")
    print("-" * 50)


if __name__ == '__main__':
    main()