import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

try:
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError as e:
    raise ImportError(
        "pytorch-grad-cam is required. Install with: pip install grad-cam"
    ) from e

from constant import MEAN, STD
from data.dataset import extract_labeled_patterned_data
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


def explain_samples(
    cam,
    metadata,
    df,
    random_seed=42,
    save_path=None,
):
    """Print GradCAM heatmaps in a 3×3 grid, one sample per class.

    Parameters
    ----------
    cam       : GradCAM  constructed externally from model + target_layer
    metadata  : dict  from load_model() (needs img_size, class_to_idx, num_classes)
    df        : pd.DataFrame  raw WM-811K dataframe with waferMap column
    random_seed : int
    save_path : str, Path, or None  saves figure if provided, else plt.show()
    """
    model = cam.model
    device = next(model.parameters()).device

    img_size = metadata["img_size"]
    class_to_idx = metadata["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = metadata["num_classes"]

    _, df_pattern, df_none = extract_labeled_patterned_data(df)

    ordered_classes = [idx_to_class[i] for i in range(num_classes)]

    n_cols = 3
    n_rows = math.ceil(num_classes / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.2, n_rows * 3.2))
    axes_flat = axes.flatten()

    for ax in axes_flat[num_classes:]:
        ax.axis("off")

    rng = np.random.default_rng(random_seed)

    for idx, class_name in enumerate(ordered_classes):
        ax = axes_flat[idx]
        pool = df_none if class_name == "none" else df_pattern
        class_pool = pool[pool["failureType"] == class_name]

        if len(class_pool) == 0:
            ax.set_title(f"{class_name}\n(no data)", fontsize=8)
            ax.axis("off")
            continue

        row = class_pool.sample(n=1, random_state=int(rng.integers(1 << 31))).iloc[0]
        tensor, rgb = preprocess_wafer(row["waferMap"], img_size, device)

        with torch.no_grad():
            logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(logits.argmax(dim=1).item())
        pred_name = idx_to_class[pred_idx]
        confidence = float(probs[0, pred_idx].item())

        grayscale_cam = cam(input_tensor=tensor)[0]
        overlay = show_cam_on_image(rgb.astype(np.float32), grayscale_cam, use_rgb=True)

        ax.imshow(overlay)
        ax.set_title(
            f"True: {class_name}\nPred: {pred_name}  ({confidence:.1%})",
            fontsize=13,
            pad=3,
        )
        ax.axis("off")

    fig.suptitle(
        f"GradCAM  |  model={metadata['model_name']}  img_size={img_size}",
        fontsize=16,
        y=1.01,
    )
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved to {save_path}")
    else:
        plt.show()
