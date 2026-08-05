import json
from pathlib import Path

import cv2
import numpy as np
import torch

from constant import MEAN, STD
from models.builder import build_model

_MEAN = np.array(MEAN, dtype=np.float32).reshape(3, 1, 1)
_STD = np.array(STD, dtype=np.float32).reshape(3, 1, 1)


def load_model(output_dir, device=None):
    """Load a trained model from an output directory.

    Parameters
    ----------
    output_dir : str or Path
        Directory produced by save_results(), containing best_model.pth
        and metadata.json.
    device : torch.device, str, or None
        Auto-selects mps > cuda > cpu when None.

    Returns
    -------
    model    : nn.Module in eval() mode on device
    metadata : dict — full contents of metadata.json
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    output_dir = Path(output_dir)
    with open(output_dir / "metadata.json") as f:
        metadata = json.load(f)

    model = build_model(
        num_classes=metadata["num_classes"],
        device=device,
        model_name=metadata["model_name"],
        pretrained=False
    )
    state = torch.load(output_dir / "best_model.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, metadata


def get_gradcam_target_layer(model, img_size, min_spatial=4):
    """Find the deepest features block whose spatial output is >= min_spatial.

    Runs a dummy forward pass through model.features to probe spatial dims,
    then scans from the last block backward.

    Parameters
    ----------
    model       : nn.Module with a model.features Sequential (EfficientNet-style)
    img_size    : list or tuple [W, H]
    min_spatial : int, minimum acceptable spatial dimension (default 4)

    Returns
    -------
    target_layer : nn.Module

    Raises
    ------
    RuntimeError if no block satisfies the constraint
    """
    if not hasattr(model, "features"):
        # Fallback: last Conv2d in the model
        for m in reversed(list(model.modules())):
            if isinstance(m, torch.nn.Conv2d):
                return m
        raise RuntimeError("Cannot find a suitable GradCAM target layer.")

    w, h = img_size[0], img_size[1]
    device = next(model.parameters()).device
    x = torch.zeros(1, 3, h, w, device=device)

    spatial_dims = []
    with torch.no_grad():
        for block in model.features:
            x = block(x)
            spatial_dims.append(x.shape[-1])

    target_idx = None
    for i in range(len(spatial_dims) - 1, -1, -1):
        if spatial_dims[i] >= min_spatial:
            target_idx = i
            break

    if target_idx is None:
        raise RuntimeError(
            f"No features block produces spatial output >= {min_spatial}. "
            f"Spatial dims: {list(enumerate(spatial_dims))}"
        )

    return model.features[target_idx]


def preprocess_wafer(wafer_map, img_size, device):
    """Convert a raw wafer map to a model-ready input tensor.

    Matches WaferDataset.__getitem__ exactly:
      resize → uint8 → stack 3ch → float32 → ImageNet normalization.

    Parameters
    ----------
    wafer_map : np.ndarray  raw 2-D wafer map, pixel values in {0, 1, 2}
    img_size  : list or tuple [W, H]
    device    : torch.device

    Returns
    -------
    tensor : torch.Tensor (1, 3, H, W) normalized, on device
    rgb    : np.ndarray (H, W, 3) float32 in [0, 1] for overlay visualization
    """
    w, h = img_size[0], img_size[1]
    resized = cv2.resize(
        np.array(wafer_map), (w, h), interpolation=cv2.INTER_NEAREST
    ).astype(np.uint8)

    img = np.stack([resized] * 3, axis=0).astype(np.float32)  # (3, H, W)
    rgb = (img / 2.0).transpose(1, 2, 0)  # (H, W, 3) in [0, 1] for overlay

    img = (img - _MEAN) / _STD
    tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    return tensor, rgb
