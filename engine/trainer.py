import copy

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
from tabulate import tabulate
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


def run_one_epoch(model, device, loader, criterion, optimizer=None, num_classes=9, threshold=None):
    """
    Run one training or evaluation epoch.

    Pass optimizer=None for eval mode (no gradient updates).
    When threshold is given (binary classification only), predict class 1
    if P(class=1) > threshold instead of argmax — useful for high-recall inference.
    Returns a dict with loss, acc, precision, recall, f1, per_class_f1, cm.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    sample_count = 0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()

        batch_n = labels.size(0)
        total_loss += loss.item() * batch_n
        sample_count += batch_n

        if threshold is not None:
            probs = torch.softmax(logits.detach(), dim=1)[:, 1]
            preds = (probs > threshold).long()
        else:
            preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    return {
        "loss":         total_loss / max(sample_count, 1),
        "acc":          accuracy_score(all_labels, all_preds),
        "precision":    precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall":       recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1":           f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "per_class_f1": f1_score(all_labels, all_preds, average=None, zero_division=0),
        "cm":           confusion_matrix(all_labels, all_preds, labels=list(range(num_classes))),
    }


def train_model(
    model,
    device,
    criterion,
    optimizer,
    scheduler,
    dataloaders,
    num_classes=9,
    epochs=12,
    unfreeze_epoch=4,
    lr=1e-3,
    backbone_lr_factor=0.1,
    should_freeze_backbone=False,
    learning_depend_metric="f1",
    epoch_callback=None,
    early_stopping=False,
    patience=10,
    min_delta=1e-4,
    class_names=None,
    checkpoint_guard=None,
    threshold=None,
):
    """
    Full training loop with optional backbone unfreeze.

    When should_freeze_backbone=True, the model's backbone should already be
    frozen and the optimizer should contain only classifier params before this
    call.  At unfreeze_epoch, backbone params are unfrozen and added as a
    second param group with lr * backbone_lr_factor.
    """
    best_state        = None
    monitor_mode      = "min" if "loss" in learning_depend_metric else "max"
    best_score        = float("inf") if monitor_mode == "min" else 0.0
    best_cm           = None
    best_f1_per_class = None
    history           = []
    patience_counter  = 0

    for epoch in range(epochs):
        print(f"===== Epoch {epoch + 1}/{epochs} =====")

        if epoch == unfreeze_epoch and should_freeze_backbone:
            print("===== Unfreezing Backbone =====")
            backbone_params = []
            for name, param in model.named_parameters():
                if "classifier" not in name and not param.requires_grad:
                    param.requires_grad = True
                    backbone_params.append(param)
            if backbone_params:
                optimizer.add_param_group({"params": backbone_params, "lr": lr * backbone_lr_factor})

        train_m = run_one_epoch(model, device, dataloaders["train"], criterion, optimizer, num_classes, threshold)
        val_m   = run_one_epoch(model, device, dataloaders["val"],   criterion, None,      num_classes, threshold)

        scheduler.step(val_m[learning_depend_metric])

        history.append({
            "epoch":          epoch,
            "train_loss":     train_m["loss"],
            "val_loss":       val_m["loss"],
            "train_f1":       train_m["f1"],
            "val_f1":         val_m["f1"],
            "train_acc":      train_m["acc"],
            "val_acc":        val_m["acc"],
            "train_precision":train_m["precision"],
            "val_precision":  val_m["precision"],
            "train_recall":   train_m["recall"],
            "val_recall":     val_m["recall"],
            "lr":             optimizer.param_groups[0]["lr"],
        })

        print_metrics(train_m, val_m)
        print_per_class_f1(val_m["per_class_f1"], class_names=class_names)
        print()

        if epoch_callback is not None:
            epoch_callback(val_m, history, epoch)

        score = val_m[learning_depend_metric]
        is_best = (score < best_score - min_delta) if monitor_mode == "min" else (score > best_score + min_delta)
        if is_best and checkpoint_guard is not None:
            is_best = checkpoint_guard(val_m, history, epoch)
        if is_best:
            best_score        = score
            best_cm           = val_m["cm"]
            best_f1_per_class = val_m["per_class_f1"]
            best_state        = copy.deepcopy(model.state_dict())
            patience_counter  = 0
        else:
            patience_counter += 1

        if early_stopping and patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1} (no improvement for {patience} epochs)")
            break

    return best_state, best_score, best_cm, best_f1_per_class, history


CLASS_NAMES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "none"]


def print_per_class_f1(per_class_f1, class_names=None):
    headers = class_names if class_names is not None else CLASS_NAMES
    rows = [[f"{v:.4f}" for v in per_class_f1]]
    print(tabulate(rows, headers=headers, tablefmt="grid"))


def print_metrics(train_m, val_m):
    headers = ["Split", "Loss", "F1", "Acc", "Precision", "Recall"]
    rows = [
        ["Train", f"{train_m['loss']:.4f}", f"{train_m['f1']:.4f}",
         f"{train_m['acc']:.4f}", f"{train_m['precision']:.4f}", f"{train_m['recall']:.4f}"],
        ["Val",   f"{val_m['loss']:.4f}",   f"{val_m['f1']:.4f}",
         f"{val_m['acc']:.4f}",   f"{val_m['precision']:.4f}",   f"{val_m['recall']:.4f}"],
    ]
    print(tabulate(rows, headers=headers, tablefmt="grid"))

def find_best_f1(metric_name, best_score, history):
    best_epoch = next(i for i in range(len(history)) if history[i][metric_name] == best_score)
    return history[best_epoch]["val_f1"]


def get_class_weights(targets, device):
    unique = np.unique(targets)
    weights = compute_class_weight(class_weight="balanced", classes=unique, y=targets)
    return torch.tensor(weights, dtype=torch.float32).to(device)

