import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import dice_loss


@torch.inference_mode()
def evaluate(net, dataloader, device, amp, class_weights=None):
    """Return dataset-level foreground metrics and the validation objective."""
    net.eval()
    loss_sum = 0.0
    pixel_count = 0
    intersection = 0
    predicted_positive = 0
    true_positive = 0

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=len(dataloader), desc='Validation round', unit='batch', leave=False):
            image = batch['image'].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = batch['mask'].to(device=device, dtype=torch.long)
            logits = net(image)

            if net.n_classes == 1:
                target = mask_true.float()
                probabilities = torch.sigmoid(logits.squeeze(1))
                loss = F.binary_cross_entropy_with_logits(logits.squeeze(1), target)
                loss += dice_loss(probabilities, target, multiclass=False)
                prediction = probabilities > 0.5
                target_bool = target.bool()
            else:
                probabilities = F.softmax(logits, dim=1).float()
                target_one_hot = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                loss = F.cross_entropy(logits, mask_true, weight=class_weights)
                loss += dice_loss(probabilities[:, 1:], target_one_hot[:, 1:], multiclass=True)
                prediction = logits.argmax(dim=1) == 1
                target_bool = mask_true == 1

            batch_pixels = mask_true.numel()
            loss_sum += loss.item() * batch_pixels
            pixel_count += batch_pixels
            intersection += (prediction & target_bool).sum().item()
            predicted_positive += prediction.sum().item()
            true_positive += target_bool.sum().item()

    # Dataset-level (micro) Dice is stable even when some validation images have no lesion.
    dice = (2.0 * intersection + 1e-6) / (predicted_positive + true_positive + 1e-6)
    precision = (intersection + 1e-6) / (predicted_positive + 1e-6)
    recall = (intersection + 1e-6) / (true_positive + 1e-6)
    net.train()
    return {
        'loss': loss_sum / max(pixel_count, 1),
        'dice': dice,
        'precision': precision,
        'recall': recall,
    }
