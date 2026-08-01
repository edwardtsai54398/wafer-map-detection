import json
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from typing import Annotated, Literal

from api.explain_service import explain_wafer as run_explain
from data.dataset import LABEL_MAP

WAFER_DATA_PATH = Path(__file__).parent / "wafer.json"
with open(WAFER_DATA_PATH) as f:
  wafer_json = json.load(f)

app = FastAPI()


@app.get("/health")
def get_health():
  return {"status": "ok"}

@app.get("/wafers")
def get_wafers():
  total_yield = 0
  total = 0
  distribution_map = {label: 0 for label in LABEL_MAP}
  for data in wafer_json:
    distribution_map[data["pred_class"]] += 1
    if data["yield"] is not None:
      total_yield += data["yield"]
      total += 1

  return {
    "yield": float(round(total_yield / total, 4)), 
    "list": wafer_json,
    "total": len(wafer_json),
    "pattern_distribution": [
      {"pred_class": k, "count": v}
      for k, v in distribution_map.items()
      ],
      "pred_class": {v+1: k for k, v in LABEL_MAP.items()}
    }


Wafer_Value = Literal[0, 1, 2]
DIM_1_List = Annotated[list[Wafer_Value], Field(min_length=1, max_length=2048)]
DIM_0_List = Annotated[list[DIM_1_List], Field(min_length=1, max_length=2048)]

class WaferRequest(BaseModel):
  wafer: DIM_0_List

  @field_validator("wafer")
  @classmethod
  def must_be_rectangular(cls, v):
    dim_1_len = len(v[0])
    for row in v:
      if(len(row) != dim_1_len):
        raise ValueError("二維陣列的每一列長度不一")
    return v

@app.post("/explain")
def explain_wafer(req: WaferRequest):

  return run_explain(req.wafer)
