# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import random


# def collate_data_and_cast(samples_list, mask_ratio_tuple, mask_probability, dtype, n_tokens=None, mask_generator=None):
#     # dtype = torch.half  # TODO: Remove
#     # print(f"Sample in collate_fn: {samples_list[0]}")
#
#     n_global_crops = len(samples_list[0][0]["global_crops"])
#     n_local_crops = len(samples_list[0][0]["local_crops"])
#
#     collated_global_crops = torch.stack([s[0]["global_crops"][i] for i in range(n_global_crops) for s in samples_list])
#
#     collated_local_crops = torch.stack([s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples_list])
#
#     B = len(collated_global_crops)
#     N = n_tokens
#     n_samples_masked = int(B * mask_probability)
#     probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
#     upperbound = 0
#     masks_list = []
#     for i in range(0, n_samples_masked):
#         prob_min = probs[i]
#         prob_max = probs[i + 1]
#         masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
#         upperbound += int(N * prob_max)
#     for i in range(n_samples_masked, B):
#         masks_list.append(torch.BoolTensor(mask_generator(0)))
#
#     random.shuffle(masks_list)
#
#     collated_masks = torch.stack(masks_list).flatten(1)
#     mask_indices_list = collated_masks.flatten().nonzero().flatten()
#
#     masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]
#
#     return {
#         "collated_global_crops": collated_global_crops.to(dtype),
#         "collated_local_crops": collated_local_crops.to(dtype),
#         "collated_masks": collated_masks,
#         "mask_indices_list": mask_indices_list,
#         "masks_weight": masks_weight,
#         "upperbound": upperbound,
#         "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
#     }

import torch
import random


def collate_data_and_cast(samples_list, mask_ratio_tuple, mask_probability, dtype, n_tokens=None, mask_generator=None):
    """
    Collate function to process the batch of samples for DINOv2 training.
    This function handles global crops, local crops, and masks for each batch.

    Args:
        samples_list: List of samples, each containing "global_crops", "local_crops", and "label".
        mask_ratio_tuple: Tuple for random masking ratios.
        mask_probability: Probability of applying masking.
        dtype: Data type for the returned tensors (e.g., torch.float32).
        n_tokens: Number of tokens per patch.
        mask_generator: Function to generate random masks.

    Returns:
        Dictionary containing batched crops, masks, and other metadata.
    """
    # Debugging: Print first sample structure to ensure data consistency
    # print(f"First sample in collate: {samples_list[0]}")

    # Process global crops
    n_global_crops = len(samples_list[0]["global_crops"])
    collated_global_crops = torch.stack(
        [sample["global_crops"][i] for i in range(n_global_crops) for sample in samples_list]
    )

    # Process local crops
    n_local_crops = len(samples_list[0]["local_crops"])
    collated_local_crops = torch.stack(
        [sample["local_crops"][i] for i in range(n_local_crops) for sample in samples_list]
    ) if n_local_crops > 0 else torch.empty(0)

    # Process labels
    labels = torch.tensor([sample["label"] for sample in samples_list], dtype=torch.long)

    # Initialize mask-related components
    B = len(collated_global_crops)  # Batch size from global crops
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []

    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
        upperbound += int(N * prob_max)
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)
    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    # Return collated batch
    return {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype) if n_local_crops > 0 else None,
        "collated_labels": labels,
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }
