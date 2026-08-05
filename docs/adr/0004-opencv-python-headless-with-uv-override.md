# ADR-0004：後端統一使用 opencv-python-headless，並以 uv override 排除 grad-cam 的 GUI 版依賴

- 狀態：已採用（Accepted）
- 日期：2026-08-05
- 影響範圍：`pyproject.toml`、`uv.lock`、`requirements.txt`、後端映像

---

## 背景

### 為什麼想用 headless 版

`opencv-python` 的 wheel 裡打包了預編譯的 OpenCV 二進位（本機 venv 實測 `site-packages/cv2/` 約 **125 MB**），其中 highgui 相關的 `.so` 會**動態連結** `libGL.so.1`、`libglib-2.0.so.0` 等系統庫。這些系統庫不在 wheel 裡，因此在 `python:3.13-slim` 這種精簡基底映像中，`import cv2` 會直接失敗：

```
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

要救就得在 Dockerfile 補一層 `apt-get install libgl1 libglib2.0-0`。

`opencv-python-headless` 是同一份原始碼關掉 highgui 重新編譯的產物：**Python API 完全相同**，只少了 `imshow`、`namedWindow`、`waitKey`、`VideoCapture` 等 GUI/視訊擷取函式，且 `.so` 不再連結那些系統庫，因此**不需要任何 apt 層**。

後端是無頭的 HTTP 服務，不存在開視窗的需求，用 headless 是自然選擇。

### 本專案實際用到的 cv2 功能

全文搜尋確認，整個 repo 對 cv2 的使用只有 resize 一項：

| 位置 | 用途 |
| --- | --- |
| `api/explain_service.py` | `cv2.resize(..., cv2.INTER_LINEAR)` 把熱力圖放大回原尺寸 |
| `explain/core.py` | `cv2.resize(..., cv2.INTER_NEAREST)` |
| `models/inference.py` | `cv2.resize(..., cv2.INTER_NEAREST)` |
| `data/dataset.py` | `cv2.resize(..., cv2.INTER_NEAREST)` |

搜尋 `imshow` 的命中全部是 matplotlib 的 `ax.imshow`（`explain/visualize.py`、`engine/visualize.py`、`train/eda.ipynb`），不是 cv2。`namedWindow`、`waitKey`、`VideoCapture`、`destroyAllWindows` 零命中。

也就是說：**訓練端和後端都不需要 GUI 版**，不只後端。

## 發生了什麼衝突

原本的寫法是把版本差異放在 group 裡：

```toml
[dependency-groups]
backend = [..., "opencv-python-headless>=4.13.0"]
train   = ["opencv-python>=4.13.0.92", ...]
```

但 `grad-cam` 在 `[project].dependencies`，而它的 `METADATA` 硬宣告依賴 `opencv-python`。結果 `uv export --group backend` 產出的 `requirements.txt` 裡**兩包同時出現**：

```
opencv-python==5.0.0.93              ← grad-cam 拉進來的
opencv-python-headless==5.0.0.93     ← 自己宣告的
```

這比「沒優化成功」更嚴重，因為兩包都把檔案裝進**同一個 `site-packages/cv2/` 目錄**（只有 `.dist-info` 目錄名不同）。後安裝的會覆寫先安裝的 `.so`，所以：

- 映像體積付了兩份錢，一份完全浪費；
- 最終生效的是 GUI 版還是 headless 版，**取決於安裝順序**，不確定；
- 若 GUI 版最後落地，而 Dockerfile 又沒有 `libgl1` 那層（目前確實沒有），`import cv2` 就會在執行期炸掉。

group 分流之所以救不了，是因為 group 決定的是「誰要裝什麼」，而衝突來自**傳遞依賴**——grad-cam 不管你在哪個 group，都會把 `opencv-python` 拉進依賴圖。

## 決策

### 1. 把 `opencv-python-headless` 提到 `[project].dependencies`

既然訓練端和後端都只用 `cv2.resize`，就沒有理由讓兩個 group 各自宣告不同的 opencv。統一成一個專案級依賴，也讓下一步的 override 不會和 group 內的宣告打架。

```toml
[project]
dependencies = [
    "numpy>=2.4.6",
    "opencv-python-headless>=4.13.0",
    "pyyaml>=6.0.3",
    "torch>=2.12.0",
    "torchvision>=0.27.0",
    "grad-cam>=1.5.5",
]

[dependency-groups]
backend = [
    "fastapi[standard]>=0.139.2",
    "pydantic>=2.13.4",
    "uvicorn[standard]>=0.51.0",
]
train = [
    "ipykernel>=7.2.0",
    "kagglehub>=1.0.1",
    "matplotlib>=3.10.9",
    ...
]
```

兩個 group 都不再出現 opencv。

### 2. 用 uv 的 `override-dependencies` 排除 `opencv-python`

```toml
[tool.uv]
# grad-cam 硬依賴 GUI 版 cv2，排除避免與 headless 重複安裝
override-dependencies = [
    "opencv-python ; platform_system == 'Never'"
]
```

概念上，uv 提供三種介入依賴解析的方式，差別在「能不能改變依賴圖的形狀」：

| 機制 | 語意 | 能移除套件嗎 |
| --- | --- | --- |
| `constraint-dependencies` | **收窄**：有人要裝這個套件時，版本必須落在範圍內 | 不能 |
| `override-dependencies` | **取代**：不論依賴圖上誰宣告了這個套件、宣告了什麼需求，一律換成這一條 | 能 |
| `[tool.uv.sources]` | **改來源**：同一個套件名改從 git／本地路徑／私有 index 取得 | 不能 |

這裡需要的是「移除」，所以只有 override 可行。做法是把需求換成一條**帶永假 marker** 的需求：`platform_system` 的值只會是 `Linux`／`Darwin`／`Windows`，`== 'Never'` 永遠不成立，於是這條需求形同不存在。

override 比對的是套件名，命中就整條替換，且作用於**依賴圖的任何一層**——這正是能攔下 grad-cam 傳遞依賴的原因。

### 3. 為什麼這樣「騙」resolver 是安全的

override 的本質是宣告「我比上游更清楚這個依賴的真實需求」，因此相容性責任轉移到本專案。這裡成立，因為 grad-cam 的宣告是**過嚴**而非必要：

- grad-cam 內部只在 `pytorch_grad_cam/utils/image.py` 使用 cv2 做 resize 與 colormap，headless 完全支援；
- 後端只 import `GradCAM` 與 `ClassifierOutputTarget`，連會用到 colormap 的 `show_cam_on_image` 都沒有碰（那個只出現在訓練路徑的 `explain/visualize.py`）；
- headless 與完整版的 Python API 相同，不存在 API 不相容問題。

## 驗證

以 uv 0.11.3 在乾淨 venv 實測（最小專案，只依賴 `grad-cam>=1.5.5`）：

| 設定 | venv 內是否有 opencv-python |
| --- | --- |
| 不加任何設定（baseline） | 有 |
| `override-dependencies` + 永假 marker | **沒有** |
| `constraint-dependencies` + 同樣 marker | 有（佐證 constraint 無法移除套件） |

## 後果

**正面**

- 依賴圖上只剩一個 cv2 提供者，`site-packages/cv2/` 不再有互相覆寫的競爭，行為變成確定的。
- 後端映像不需要 `apt-get install libgl1 libglib2.0-0`。這才是主要收益——省下的是一整套 X11/Mesa 系統庫與一層 apt cache，而不只是 wheel 體積。
- 訓練與後端共用同一份 opencv 宣告，少一個環境差異來源。

**代價與注意事項**

- override 是全域且凌駕一切的機制。日後若有其他套件真的需要 GUI 版 opencv，這條設定會**靜默地**把它也擋掉，症狀是執行期缺函式而非解析期報錯。
- `requirements.txt` 中仍會保留一行帶 marker 的條目：

  ```
  opencv-python==5.0.0.93 ; platform_system == 'Never'
  ```

  這是預期行為，pip 會求值 marker 後跳過。**驗證是否真的沒裝要查 site-packages，不要查 requirements.txt。**

- pip 本身沒有 override 的概念，因此 override 必須在產生 `requirements.txt` 的 `uv export` 階段就套用完畢，Dockerfile 的 `pip install -r requirements.txt` 無從補救。目前流程正是如此，相容。
- 需重跑 lock 與 export：

  ```
  uv lock
  uv export --group backend --format requirements-txt --no-dev --no-emit-project --output-file requirements.txt
  ```

  部署前建議驗一次映像內只剩一個 opencv：

  ```
  docker run --rm <image> sh -c "pip list | grep -i opencv"
  ```

## 曾考慮的其他方案

**放棄 headless，接受 grad-cam 帶進來的 `opencv-python`**
可行但代價是映像多背一層 apt 系統庫與重複的二進位，只為了滿足一個本專案永遠不會呼叫的 GUI 依賴宣告。

**尋找不依賴 `opencv-python` 的 grad-cam 版本**
PyPI 上 `grad-cam` 只有一個官方發行版，沒有 headless variant。改用社群 fork 為了單一依賴宣告引入供應鏈與維護風險，不划算。

