import torch
import torch.nn.functional as F
from tqdm import tqdm

from train_utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                # 修复: 找出所有值为 255 的无效区域
                ignore_mask = (mask_true == 255)
                # 为了防止 F.one_hot 崩溃，先将 255 临时设为 0
                mask_true[ignore_mask] = 0

                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true_onehot = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred_onehot = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()

                # 修复: 将无效区域的 one_hot 值全部清零，使其不参与 Dice 计算
                ignore_mask = ignore_mask.unsqueeze(1).expand_as(mask_true_onehot)
                mask_true_onehot[ignore_mask] = 0
                mask_pred_onehot[ignore_mask] = 0

                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred_onehot[:, 1:], mask_true_onehot[:, 1:],
                                                    reduce_batch_first=False)
    net.train()
    return dice_score / max(num_val_batches, 1)