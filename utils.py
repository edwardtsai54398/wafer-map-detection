import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from constant import DEFAULT_MODEL_NAME
from engine.visualize import plot_cm, plot_history


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EpochTimer:
    def __init__(self):
        self._times = []
        self._last = time.time()

    def __call__(self, val_m, history, epoch):
        now = time.time()
        self._times.append(now - self._last)
        self._last = now

    def get_time_per_epoch(self):
        if not self._times:
            return 0.0
        avg = sum(self._times) / len(self._times)
        print(f"Average time per epoch: {avg:.1f}s")
        return avg


def make_output_dir(experiment_name, base="outputs"):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(base) / f"{experiment_name}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_results(out_dir, model, best_state, best_val_f1, test_f1, test_cm, test_f1_per_class,
                history, num_classes, label_map, image_size,train_time_per_epoch=0.0, inference_time=0.0, model_name=DEFAULT_MODEL_NAME):
    model.load_state_dict(best_state)

    # 找出best_epoch
    best_epoch = next(i for i in range(len(history)) if history[i]["val_f1"] == best_val_f1)

    # 參數量
    num_parameters = sum(p.numel() for p in model.parameters()) / 1e6  # 單位：M

    # 過擬合差距
    val_train_loss_gap = history[best_epoch]["val_loss"] - history[best_epoch]["train_loss"]


    torch.save(best_state, out_dir / "best_model.pth")

    metadata = {
        "model_name":   model_name,
        "num_parameters": num_parameters,
        "num_classes":  num_classes,
        "class_to_idx": label_map,
        "img_size":     list(image_size),
        "best_epoch": best_epoch,
        "best_val_f1":  best_val_f1,
        "test_f1": test_f1,
        "test_f1_per_class": test_f1_per_class.tolist() if test_f1_per_class is not None else [],
        "train_time_per_epoch": train_time_per_epoch,
        "inference_time": inference_time,
        "val_train_loss_gap": val_train_loss_gap
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=4)

    history_fig = plot_history(history)
    history_fig.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close(history_fig)

    if test_cm is not None:
        cm_fig = plot_cm(test_cm, num_classes)
        cm_fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
        plt.close(cm_fig)

    print(f"Results saved to {out_dir}")
