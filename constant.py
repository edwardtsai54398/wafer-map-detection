LABEL_MAP = {
    "Center": 0,
    "Donut": 1,
    "Edge-Loc": 2,
    "Edge-Ring": 3,
    "Loc": 4,
    "Near-full": 5,
    "Random": 6,
    "Scratch": 7,
    "none": 8,
}

PATTERN_LABEL_MAP = {k: v for k, v in LABEL_MAP.items() if k != "none"}

NONE_NONONE_LABEL_MAP = {"none": 0, "non-none": 1}

IDX_TO_CLASS = {v: k for k, v in LABEL_MAP.items()}

CLASS_NAMES = list(LABEL_MAP.keys())

# -------------------------------------------------------- preprocessing ----

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# --------------------------------------------------------------- models ----

MODELS = {
    "EFFICIENTNET_B4": "efficientnet_b4",
    "EFFICIENTNET_B2": "efficientnet_b2",
    "EFFICIENTNET_B0": "efficientnet_b0",
}

DEFAULT_MODEL_NAME = MODELS["EFFICIENTNET_B4"]
