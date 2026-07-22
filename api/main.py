import json
from pathlib import Path
from fastapi import FastAPI, UploadFile

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
