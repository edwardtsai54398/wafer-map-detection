import torch.nn as nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights, efficientnet_b2, EfficientNet_B2_Weights, efficientnet_b0, EfficientNet_B0_Weights

from constant import DEFAULT_MODEL_NAME, MODELS

# model name → (constructor, pretrained weights)
_MODEL_REGISTRY = {
    MODELS["EFFICIENTNET_B4"]: (efficientnet_b4, EfficientNet_B4_Weights.IMAGENET1K_V1),
    MODELS["EFFICIENTNET_B2"]: (efficientnet_b2, EfficientNet_B2_Weights.IMAGENET1K_V1),
    MODELS["EFFICIENTNET_B0"]: (efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1),
}


def replace_head(model, num_classes):
    """Replace the final Linear layer with a new one sized for num_classes."""
    last_layer_name, last_layer = list(model.named_modules())[-1]
    parent = model
    for part in last_layer_name.split(".")[:-1]:
        parent = getattr(parent, part)
    attr = last_layer_name.split(".")[-1]
    setattr(parent, attr, nn.Linear(last_layer.in_features, num_classes))
    return model


def build_model(num_classes, device, model_name=DEFAULT_MODEL_NAME):
    """Return a model with a custom classification head on device."""
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(_MODEL_REGISTRY)}")
    model_fn, weights = _MODEL_REGISTRY[model_name]
    model = model_fn(weights=weights)
    model = replace_head(model, num_classes)
    return model.to(device)
