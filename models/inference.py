import torch
import torch.nn.functional as F
import cv2
import numpy as np

from constant import IDX_TO_CLASS, MEAN, STD

_MEAN = np.array(MEAN, dtype=np.float32).reshape(3, 1, 1)
_STD = np.array(STD, dtype=np.float32).reshape(3, 1, 1)

# stage-1 non-none probability threshold for two-stage inference
TWO_STAGE_THRESHOLD = 0.3


@torch.no_grad()
def predict(model, tensor, idx_to_class=None):
    """Single-stage inference.

    Parameters
    ----------
    model        : nn.Module  in eval() mode on device
    tensor       : torch.Tensor  (B, 3, H, W) on the same device as model
    idx_to_class : dict[int, str] or None  maps class index → class name

    Returns
    -------
    class_indices : LongTensor (B,)        predicted class index per sample
    class_names   : list[str] (len B)      predicted class name ('' if idx_to_class is None)
    scores        : FloatTensor (B, C)     per-class softmax confidence, scores[i, j] = P(class j | sample i)
    """
    was_training = model.training
    model.eval()

    logits = model(tensor)
    scores = F.softmax(logits, dim=1)
    class_indices = scores.argmax(dim=1)

    if was_training:
        model.train()

    if idx_to_class is not None:
        class_names = [idx_to_class[int(i)] for i in class_indices]
    else:
        class_names = [""] * len(class_indices)

    return class_indices, class_names, scores


@torch.no_grad()
def predict_two_stage(model_s1, model_s2, tensor, threshold=TWO_STAGE_THRESHOLD):
    """Two-stage inference for none vs. pattern classification.

    Stage 1 is a binary model (none=0, non-none=1).
    Stage 2 is an 8-class pattern model (Center=0 … Scratch=7).
    Combined output covers all 9 classes in LABEL_MAP order
    (Center=0 … Scratch=7, none=8).

    Combined scores are formed as:
      scores[:, 0:8] = p_nonnone_s1 * p_s2   (marginalised pattern probs)
      scores[:, 8]   = p_none_s1              (none probability)

    This gives a proper distribution that sums to 1.

    Parameters
    ----------
    model_s1  : nn.Module  Stage 1 binary model in eval() mode
    model_s2  : nn.Module  Stage 2 8-class model in eval() mode
    tensor    : torch.Tensor  (B, 3, H, W) on the same device as both models
    threshold : float or None
        Stage 1 non-none threshold.
        - If float: samples with p(non-none) > threshold → run S2 argmax; else class=8
        - If None:  argmax on combined 9-class scores

    Returns
    -------
    class_indices : LongTensor (B,)       predicted class index (0-8, 8=none)
    class_names   : list[str] (len B)     predicted class name from LABEL_MAP
    scores        : FloatTensor (B, 9)    per-class confidence for all 9 classes
    """
    s1_was_training = model_s1.training
    s2_was_training = model_s2.training

    model_s1.eval()
    model_s2.eval()

    p_s1 = F.softmax(model_s1(tensor), dim=1)  # (B, 2)
    p_s2 = F.softmax(model_s2(tensor), dim=1)  # (B, 8)

    # Combined 9-class probability distribution
    scores = torch.empty(tensor.size(0), 9, dtype=p_s1.dtype, device=p_s1.device)
    scores[:, :8] = p_s1[:, 1:2] * p_s2   # p(non-none) × p(pattern_k)
    scores[:, 8] = p_s1[:, 0]              # p(none)

    if threshold is not None:
        is_nonnone = p_s1[:, 1] > threshold
        class_indices = torch.full((tensor.size(0),), fill_value=8, dtype=torch.long, device=p_s1.device)
        if is_nonnone.any():
            class_indices[is_nonnone] = p_s2[is_nonnone].argmax(dim=1)
    else:
        class_indices = scores.argmax(dim=1)

    class_names = [IDX_TO_CLASS[int(i)] for i in class_indices]

    # restore training state
    if s1_was_training:
        model_s1.train()
    if s2_was_training:
        model_s2.train()

    return class_indices, class_names, scores


def wafer_to_tensor(wafer, image_size=(64, 64), device=None):
    # resize
    wafer = cv2.resize(wafer, image_size, interpolation=cv2.INTER_NEAREST)

    # 3 kernels
    wafer = np.stack([wafer] * 3, axis=0).astype(np.float32)

    # normalization
    wafer = (wafer - _MEAN) / _STD
    
    tensor = torch.from_numpy(wafer)
    return tensor