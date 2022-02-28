import torch
from torch import Tensor


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    # Dice = 2 * TP / (2 * TP + FP + FN)
    # https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient#Formula
    assert input.size() == target.size()
    if input.dim() == 3 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 3 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def dice_loss(input: Tensor, target: Tensor):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    return 1 - dice_coeff(input, target, reduce_batch_first=True)


def jaccard_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Jaccard coefficient for all batches, or for a single mask
    # Jaccard = TP / ( TP + FP + FN )
    # https://en.wikipedia.org/wiki/Jaccard_index#Similarity_of_asymmetric_binary_attributes
    assert input.size() == target.size()
    if input.dim() == 3 and reduce_batch_first:
        raise ValueError(f'Jaccard: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 3 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = inter

        return (inter + epsilon) / (sets_sum - inter + epsilon)
    else:
        # compute and average metric for each batch element
        jaccard = 0
        for i in range(input.shape[0]):
            jaccard += dice_coeff(input[i, ...], target[i, ...])
        return jaccard / input.shape[0]



def jaccard_loss(input: Tensor, target: Tensor):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    return 1 - jaccard_coeff(input, target, reduce_batch_first=True)