# ADR-0003：移除未使用的 pyyaml 依賴

- 狀態：已採用（Accepted）
- 日期：2026-08-05
- 影響範圍：`pyproject.toml`、`uv.lock`、`requirements.txt`、後端映像

---

## 背景

`pyyaml` 從第一版 `pyproject.toml`（`930ad63 fix: uv init`）就列在 `[project].dependencies` 裡，和 `numpy`、`torch`、`torchvision` 一起被加進去。那個 commit 只有 `pyproject.toml`、`.python-version`、`uv.lock` 三個檔案——**專案還沒有任何程式碼**。也就是說它是在規劃階段預先列上的，預期會有某種 YAML 設定檔（資料集定義、訓練超參數之類的），而不是為了讓某段程式跑起來。

那份設定檔後來沒有出現。訓練用 notebook 的 constant 寫在 `constant.py`，模型資訊存成 `metadata.json`，API 沒有任何設定檔。

現在的事實是：

1. **沒有任何程式碼 import 它**。`api/`、`explain/`、`models/`、`train/*.ipynb` 全文搜尋 `import yaml` / `from yaml` / `yaml.` 都是零命中。
2. **也沒有其他套件需要它**：

   ```
   $ uv tree --invert --package pyyaml
   pyyaml v6.0.3
   └── wafer-map-detection v0.1.0
   ```

   只有本專案宣告它。常見的誤會是 `uvicorn[standard]` 會拉進 PyYAML——舊版確實會，但目前 lock 到的版本已經不含它了。`grad-cam`、`kagglehub`、`fastapi[standard]` 也都沒有依賴。

因為它掛在 `[project].dependencies` 而不是某個 group，`uv export --group backend` 會把它寫進 `requirements.txt`，於是它也進了後端映像。

## 決策

從 `[project].dependencies` 移除 `pyyaml`，**不移到 dev 或其他 group**。

移到 dev 的前提是「開發時會用到、部署時不用」；但這裡是開發時也沒用到，任何 group 都不是它的位置。等到真的有 YAML 設定檔要讀，再依當時的使用位置決定加在哪一組。

`docker-compose.yml` 是 Docker 自己解析的，跟 Python 有沒有 pyyaml 無關。

## 什麼情況要加回來

- 用 YAML 檔當 uvicorn 的 `--log-config`（目前 Dockerfile、docker-compose.yml、CMD 都沒有這個參數）。
- 引入任何自己寫的 YAML 設定檔。
- 未來的依賴需要它——那種情況它會是間接依賴，由 `uv.lock` 自己處理，不需要顯式宣告。

## 後果

**正面**

- 依賴清單反映真實使用狀況，`pyproject.toml` 不再有「不知道為什麼在這裡」的項目。
- `requirements.txt` 與後端映像少一個套件。

**代價**

- 節省的體積很小（pyyaml 約數百 KB，相對於 torch 幾乎可忽略），這個決策的價值主要在可維護性而不是映像大小。
- 需要重跑 `uv lock` 並重新產出 `requirements.txt`：

  ```
  uv lock
  uv export --group backend --format requirements-txt --no-dev --no-emit-project --output-file requirements.txt
  ```
