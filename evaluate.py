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
    denominator = predicted_positive + true_positive
    dice = 1.0 if denominator == 0 else 2.0 * intersection / denominator
    # Do not use epsilon here: if no positive pixel was predicted, precision
    # is not 1.0; reporting 0 makes the failure mode unambiguous.
    precision = 0.0 if predicted_positive == 0 else intersection / predicted_positive
    recall = 0.0 if true_positive == 0 else intersection / true_positive
    net.train()
    return {
        'loss': loss_sum / max(pixel_count, 1),
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'predicted_foreground_ratio': predicted_positive / max(pixel_count, 1),
        'true_foreground_ratio': true_positive / max(pixel_count, 1),
    }
