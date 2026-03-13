import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class IDRiDDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(IDRiDDataset, self).__init__()

        # 根据 train 布尔值，匹配官方数据集的奇葩文件夹名称
        self.split_name = "a. Training Set" if train else "b. Testing Set"

        # 官方原图目录
        self.img_dir = os.path.join(root, "1. Original Images", self.split_name)
        # 官方病灶掩膜目录
        self.gt_dir = os.path.join(root, "2. All Segmentation Groundtruths", self.split_name)

        assert os.path.exists(self.img_dir), f"Image path '{self.img_dir}' does not exist. Please check data_path."
        assert os.path.exists(self.gt_dir), f"Groundtruth path '{self.gt_dir}' does not exist."

        # 保持最初的 transforms
        self.transforms = transforms

        # 1. 读取原图 (.jpg)
        img_names = [i for i in os.listdir(self.img_dir) if i.endswith(".jpg")]
        self.img_list = [os.path.join(self.img_dir, i) for i in img_names]

        # 2. 构建 4 种病灶掩膜的路径字典，严格按照官方文件夹名称
        self.mask_paths = []
        for img_name in img_names:
            base_name = img_name.replace(".jpg", "")
            masks_for_this_img = {
                "MA": os.path.join(self.gt_dir, "1. Microaneurysms", f"{base_name}_MA.tif"),
                "HE": os.path.join(self.gt_dir, "2. Haemorrhages", f"{base_name}_HE.tif"),
                "EX": os.path.join(self.gt_dir, "3. Hard Exudates", f"{base_name}_EX.tif"),
                "SE": os.path.join(self.gt_dir, "4. Soft Exudates", f"{base_name}_SE.tif")
            }
            self.mask_paths.append(masks_for_this_img)

    def __getitem__(self, idx):
        # 1. 读取原图，并直接 Resize 到 800x800
        img = Image.open(self.img_list[idx]).convert('RGB')
        img = img.resize((800, 800), Image.BILINEAR)

        img_np = np.array(img)
        h, w = img_np.shape[:2]

        # 2. 动态生成 ROI Mask (在已经缩放为 800x800 的图上提取)
        red_channel = img_np[:, :, 0]
        # 提取视野外部纯黑背景 (设为 255 以在 Loss 中忽略)
        roi_mask = np.where(red_channel > 10, 0, 255).astype(np.uint8)

        # 3. 合并多分类 Mask
        final_mask = np.zeros((h, w), dtype=np.uint8)

        paths = self.mask_paths[idx]
        # 类别映射
        class_mapping = {
            "MA": 1,  # 微动脉瘤
            "HE": 2,  # 出血
            "EX": 3,  # 硬性渗出
            "SE": 4  # 软性渗出
        }

        # 依次读取各个病灶的掩膜并填入 final_mask
        for cls_name, class_idx in class_mapping.items():
            mask_path = paths[cls_name]
            # 判断掩膜文件是否存在（兼容缺失某类病灶的图片）
            if os.path.exists(mask_path):
                cls_mask = Image.open(mask_path).convert('L')

                # 掩膜 Resize 必须使用最近邻插值(NEAREST)！
                cls_mask = cls_mask.resize((800, 800), Image.NEAREST)

                cls_mask_np = np.array(cls_mask)
                final_mask[cls_mask_np > 0] = class_idx

        # 4. 将生成的 ROI 黑边强行覆盖上去
        final_mask = np.where(roi_mask == 255, 255, final_mask)
        mask = Image.fromarray(final_mask)

        # 5. 执行数据增强和预处理
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs