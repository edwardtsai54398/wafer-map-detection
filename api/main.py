import json
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from typing import Annotated, Literal

from api.explain_service import explain_wafer as run_explain

WAFER_DATA_PATH = Path(__file__).parent / "wafer.json"
with open(WAFER_DATA_PATH) as f:
  wafer_json = json.load(f)

app = FastAPI()


@app.get("/health")
def get_health():
  return {"status": "ok"}

@app.get("/wafers")
def get_wafers():
  return {"list": wafer_json}


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
