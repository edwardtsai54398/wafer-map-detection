import sys
from pathlib import Path
import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
sys.path.insert(0, '..')

from constant import LABEL_MAP
from models.inference import predict_two_stage
from explain import load_model, preprocess_wafer, get_gradcam_target_layer

TWO_STAGE_MODEL_PATH = Path(__file__).parent.parent / "outputs/two_stage_20260531_110545"

DEVICE = torch.device("cpu")
model_s1, metadata_s1 = load_model(TWO_STAGE_MODEL_PATH/"stage1", DEVICE)
model_s2, metadata_s2 = load_model(TWO_STAGE_MODEL_PATH/"stage2", DEVICE)

target_layer_s1 = get_gradcam_target_layer(model_s1, metadata_s1["img_size"])
target_layer_s2 = get_gradcam_target_layer(model_s2, metadata_s2["img_size"])

cam_s1 = GradCAM(model=model_s1, target_layers=[target_layer_s1])
cam_s2 = GradCAM(model=model_s2, target_layers=[target_layer_s2])

def explain_wafer(wafer_2d):
  wafer = np.array(wafer_2d)
  image_size = metadata_s1["img_size"]
  tensor, _ = preprocess_wafer(wafer, image_size, DEVICE)

  # inference
  class_indices, class_names, scores = predict_two_stage(model_s1, model_s2, tensor)

  final_idx = int(class_indices[0])
  final_name = str(class_names[0])
  confidence = float(scores[0][final_idx])

  # grad-cam
  if(final_idx == LABEL_MAP["none"]):
    cam = cam_s1
    target = metadata_s1["class_to_idx"][final_name]
  else:
    cam = cam_s2
    target = metadata_s2["class_to_idx"][final_name]

  grayscale = cam(tensor, targets=[ClassifierOutputTarget(int(target))])[0]

  # resize to original
  orig_h, orig_w = len(wafer), len(wafer[0])
  resized_grayscale = cv2.resize(grayscale, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

  return {
    "prediction": {
      "class_index": final_idx,
      "class_name": final_name,
      "confidence": round(confidence, 4),
    },
    "scores": sorted(
    [
        {"id": v+1, "class": k, "score": round(float(scores[0][v]), 4)}
        for k, v in LABEL_MAP.items()
    ],
    key=lambda x: x["score"],
    reverse=True,   # 由大到小；升冪就拿掉
),
    "explanation": {
      "heat_map": np.round(resized_grayscale.astype(np.float64), 2).tolist(),
      "height": orig_h,
      "width": orig_w
    }
  }
