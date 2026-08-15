# ADR-0001：後端以 CPU-only PyTorch 部署，容器建置直接用 `uv sync`

- 狀態：已採用（Accepted）
- 日期：2026-08-05（決策一已於 2026-08-15 更新，見下）
- 影響範圍：`pyproject.toml`、`uv.lock`、`Dockerfile`、`docker-compose.yml`

---

## 背景

這個 repo 同時承擔兩種用途：

1. **訓練**：在本機（macOS）或 GPU 機器上跑 notebook，需要 matplotlib、pandas、scikit-learn、ipykernel 等一整套分析工具。
2. **後端推論**：`api/main.py` 的 FastAPI 服務，要打包成容器部署，只做 forward pass 與 Grad-CAM，**不需要 CUDA**。

如果兩者共用同一組依賴，後端映像會被迫拉進預設的 PyPI `torch`，而它會連帶安裝 `nvidia-cublas`、`nvidia-cudnn` 等 CUDA wheel，體積是 GB 等級，對一個純 CPU 推論服務完全是浪費。

同時要決定映像怎麼安裝依賴。FastAPI 官方 Docker 文件的範例是 `pip install -r requirements.txt`，而 uv 提供 `uv export` 可以從 lock 產出 requirements.txt，看起來是自然的接法。

## 決策

### 決策一：依 dependency group 切開 train / backend，backend 的 torch 綁到 CPU-only index

`pyproject.toml` 中：

- 分成 `backend`（fastapi、uvicorn、torch、torchvision、grad-cam）與 `train`（notebook 與分析工具 + torch、torchvision、grad-cam）兩個 group。
- 宣告 explicit index `pytorch-cpu = https://download.pytorch.org/whl/cpu`，並在 `[tool.uv.sources]` 中把 `torch` / `torchvision` 指向該 index。
- 附帶：`grad-cam` 硬依賴 GUI 版 `opencv-python`，用 `override-dependencies` 把它排除，避免與專案本身的 `opencv-python-headless` 重複安裝。

> **2026-08-15 更新**：`pytorch-cpu` index 原本只綁定 `backend` group，`train` group 的 torch 走預設 PyPI 來源（保留 GPU 機器跑 notebook 的 CUDA 能力）。因為同一套件在兩個 group 來自不同 registry，uv 無法在同一次解析中共存，當時另外宣告了 `conflicts = [[{group = "backend"}, {group = "train"}]]`，代價是兩個 group 不能同時 `sync`，開發時要碰兩邊程式碼（例如 `explain_service.py`）得切換 venv。
>
> 現在把 `train` 也改綁 `pytorch-cpu`，兩個 group 來源一致，`conflicts` 隨之移除，可以 `uv sync --group backend --group train` 用同一份 venv 開發。代價是 **`train` group 在 Linux/Windows 上失去 CUDA 支援**——[base_train.ipynb](../../train/base_train.ipynb) 等 notebook 裡 `torch.cuda.is_available()` 會恆為 `False`。macOS 不受影響（`pytorch-cpu` index 對 `sys_platform == 'darwin'` 解析出的仍是一般版 `torch`，MPS 照常可用）。需要 CUDA 訓練時，仍可用預設 PyPI 來源另外裝一份 GPU 版 torch（例如非 uv 管理的環境，或 Kaggle/Colab 這類雲端 GPU 筆記本），本 repo 的 `uv sync --group train` 僅保證 CPU 可跑。

### 決策二：容器內直接 `uv sync`，不產生也不 commit requirements.txt

`Dockerfile` 的流程是：

```dockerfile
RUN pip install uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --group backend
COPY . /app
CMD [".venv/bin/uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

repo 內**不保留 requirements.txt**（開發流程一律用 `uv sync`，用不到它）。

## 關鍵理由：`uv export` 產出的 requirements.txt 不帶 index 資訊

這是走 requirements.txt 路線時踩到的地雷。用

```
uv export --group backend --format requirements-txt --no-dev --no-emit-project
```

（uv 0.11.3 實測）產出的檔案裡，torch 那幾行長這樣：

```
torch==2.13.0 ; sys_platform == 'darwin'
torch==2.13.0+cpu ; sys_platform != 'darwin'
torchvision==0.28.0+cpu ; sys_platform != 'darwin'
```

檔案裡**沒有任何 `--index-url` / `--extra-index-url`**。也就是說「這些 wheel 來自 `download.pytorch.org/whl/cpu`」這個資訊在 export 的當下就被丟掉了。

Linux 容器命中的是 `+cpu` 那行，pip 只會去 PyPI 找，而 PyPI 上根本不存在 `2.13.0+cpu` 這個 local version → `RUN pip install -r requirements.txt` **必定失敗**。

要救就得在 Dockerfile 自己補上 `--extra-index-url https://download.pytorch.org/whl/cpu`。那等於把「套件來源」這個本來寫在 `pyproject.toml` 的事實，手抄一份到 Dockerfile；日後改 index 就有兩個地方要同步，而且漏改的症狀是 build 失敗或裝到錯的 wheel。

更麻煩的是 export 預設帶 `--hash`。補了 `--extra-index-url` 之後，pip 對同名同版本套件在多個 index 之間的取捨是不保證的（例如 macOS 分支的 `torch==2.13.0` 無 local version，PyPI 與 PyTorch index 都有，但 hash 不同），一旦抓錯來源就是 hash mismatch。

`uv sync --frozen` 沒有這個問題：它直接讀 `uv.lock`，lock 裡每個套件都記錄了自己的 `source = { registry = ... }`，來源與版本是同一份事實。

## 考慮過的其他方案

| 方案 | 為什麼不採用 |
|---|---|
| A. 在 repo 產生並 commit requirements.txt，Docker `pip install -r` | 多一份需要手動同步的衍生產物（已經有 uv.lock 了），且有上述 index 遺失問題；開發流程也用不到這個檔案。 |
| B. 在容器內先 `uv export` 再 `pip install` | index 遺失問題完全一樣，沒有改善；還多一層「uv 解析完再交給 pip 重裝」的繞路。 |
| C. Dockerfile 裡先 `pip install torch --index-url .../cpu`，其餘依賴另外裝 | 版本與來源都手抄在 Dockerfile，與 `uv.lock` 脫鉤，是最容易長期漂移的做法。 |
| D. **`uv sync --frozen --no-dev --group backend`（採用）** | — |

## 後果

**正面**

- 依賴的單一事實來源就是 `pyproject.toml` + `uv.lock`，沒有第二份需要同步的清單。
- `--frozen` 保證容器裝到的版本與本機解析結果完全一致，build 時不會偷偷重新解析。
- 後端映像不含任何 `nvidia-*` CUDA wheel。
- `COPY pyproject.toml uv.lock` 早於 `COPY . /app`，改應用程式碼不會讓依賴層失效。

**代價與已知待辦**

- 映像需要 uv 本身。目前用 `RUN pip install uv`，**版本沒有鎖定**，build 結果不是完全可重現；之後可改成 `COPY --from=ghcr.io/astral-sh/uv:<version>` 或 pin 版本。
- `uv sync` 產出的是 `/app/.venv`，所以 `CMD` 必須寫 `.venv/bin/uvicorn`（或改設 `ENV PATH="/app/.venv/bin:$PATH"`）。
- 目前是單階段 build，uv 與 uv cache 都留在最終映像裡，尚未做 multi-stage 瘦身。
- ~~`train` 與 `backend` 宣告為 conflicts，代表不能同時 sync 兩個 group；本機訓練與後端開發要切換環境。~~ 已於 2026-08-15 解除（見上方決策一更新），代價轉為 `train` group 在 Linux/Windows 上無法使用 CUDA。

## 相關

- FastAPI 官方 Docker 部署文件：https://fastapi.tiangolo.com/deployment/docker/
- uv indexes / conflicting groups：https://docs.astral.sh/uv/concepts/indexes/
