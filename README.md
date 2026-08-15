# Wafer Map Defect Detection

基於 EfficientNet-B4 遷移學習的晶圓圖缺陷模式分類。支援單階段（9 類）與兩階段（先判斷有無缺陷，再分類缺陷模式）兩種訓練模式。

資料集：[WM-811K Wafer Map](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)（透過 `kagglehub` 自動下載）

---

## 專案結構

```
wafer-map-detection/
├── eda.ipynb                          # 資料探索：分佈分析、缺陷模式視覺化
├── base_train.ipynb                   # 單階段基線訓練（9 類，以最佳 val F1 儲存）
├── base_loss_train.ipynb              # 單階段訓練（以 loss 最小化為目標）
├── with_class_weights_train.ipynb     # 單階段訓練（類別加權 + none undersampling）
├── two_stage_train.ipynb              # 兩階段訓練（二元偵測 → 缺陷模式分類）
├── constant.py             # 跨模組共用常數（零 import）：LABEL_MAP、MEAN/STD、MODELS
├── data/
│   └── dataset.py          # WaferDataset、extract_labeled_patterned_data
├── models/
│   └── builder.py          # replace_head、build_model
├── engine/
│   ├── trainer.py          # run_one_epoch、train_model、print_metrics
│   └── visualize.py        # plot_history、plot_cm
├── api/
│   ├── main.py             # FastAPI 入口：/health、/wafers、/explain
│   ├── explain_service.py  # 兩階段推論 + Grad-CAM 熱力圖
│   └── wafer.json          # /wafers 使用的預先推論結果
├── configs/                           # train.py CLI 用的 YAML 設定（備用）
├── train.py                           # CLI 入口（備用；目前以 notebook 為主）
└── outputs/                # 每次實驗自動建立的輸出目錄
    └── {experiment}_{timestamp}/
        ├── best_model.pth
        ├── metadata.json
        ├── training_curves.png
        └── confusion_matrix.png
```

---

## 環境安裝

依賴以 [uv](https://docs.astral.sh/uv/) 管理，切成 `backend`（FastAPI 服務）與 `train`（notebook 訓練）兩個 dependency group，皆使用 CPU-only 版 PyTorch，可視需要單獨或同時安裝：

```bash
uv sync --group backend            # 只跑 API
uv sync --group train              # 只跑訓練 notebook
uv sync --group backend --group train   # 開發時兩者都要（同一份 venv）
```

> 需要 Kaggle API Token 才能透過 `kagglehub` 下載資料集。至 Kaggle 網站 Settings → API → Create New Token，依指示執行：
> ```bash
> mkdir -p ~/.kaggle
> echo "<your_token>" > ~/.kaggle/access_token
> chmod 600 ~/.kaggle/access_token
> ```

---

## 訓練方式

所有訓練實驗均以 Jupyter Notebook 進行。超參數直接在 notebook 頂部的變數區塊設定，不需要 YAML config。

### 資料探索

先執行 `eda.ipynb` 了解資料分佈、各缺陷模式的樣本數與視覺特徵，以及最佳 resize 目標尺寸的推導過程。

### 訓練 Notebooks

| Notebook | 模式 | 策略 | 主要超參數 |
|---|---|---|---|
| `base_train.ipynb` | single（9 類） | 無特殊平衡；以最佳 val F1 儲存 checkpoint | 64×64, batch 32, 24 epochs, LR 0.001 |
| `base_loss_train.ipynb` | single（9 類） | ReduceLROnPlateau `mode="min"`，以 loss 最小化為存檔依據 | 32×32, batch 32, 24 epochs, LR 0.001 |
| `with_class_weights_train.ipynb` | single（9 類） | sklearn `compute_class_weight` 加權損失 + none 類別上限 10,000 筆 | 32×32, batch 32, 24 epochs, LR 0.001 |
| `two_stage_train.ipynb` | two_stage | Stage 1：二元分類（有無缺陷），自訂判定閾值 0.3 提高召回率；Stage 2：8 類缺陷模式分類 | 32×32, batch 32, 各 24 epochs, LR 0.001 |

---

## 類別說明

### 單階段 Label Map（9 類）

| Index | 類別 |
|---|---|
| 0 | Center |
| 1 | Donut |
| 2 | Edge-Loc |
| 3 | Edge-Ring |
| 4 | Loc |
| 5 | Near-full |
| 6 | Random |
| 7 | Scratch |
| 8 | none（無缺陷） |

### 兩階段

- **Stage 1**：`0 = no_defect`，`1 = defect`
- **Stage 2**：同上表的 Index 0–7（排除 none）

---

## 實驗結果摘要

### 各實驗 Val F1 概覽

| 實驗 | 圖像大小 | Val F1 (macro) | Scratch F1 | 備注 |
|---|---|---|---|---|
| `baseline` 32×32 (`20260525`) | 32×32 | 0.8218 | 0.5371 | 各類表現均衡，none 最高 (0.9831) |
| `baseline` 64×64 (`20260530`) | 64×64 | **0.8656** | 0.7358 | 圖像放大後整體提升，Scratch 改善最明顯 |
| `class_weights` run 1 (`20260525_171253`) | 32×32 | 0.6667 | 0.1046 | 加權策略嚴重惡化 Scratch |
| `class_weights` run 2 (`20260525_194832`) | 32×32 | 0.7291 | 0.2156 | 略有改善但 Scratch 仍遠低於基線 |
| `base_loss` run 1 (`20260527_142536`) | 32×32 | 0.1341 | 0.3077 | val_f1 異常低（疑似 checkpoint 儲存問題） |
| `base_loss` run 2 (`20260527_223003`) | 32×32 | 0.7299 | 0.3077 | Edge-Ring 最高 (0.9621)，Scratch 仍為瓶頸 |
| `two_stage` Stage 1 (`20260531`) | 64×64 | 0.9582 | — | 二元分類：none 0.9872 / non-none 0.9291 |
| `two_stage` Stage 2 (`20260531`) | 64×64 | **0.9075** | **0.8701** | 8 類缺陷分類，Scratch 大幅提升 |

### 各類 F1 細節（主要實驗）

| 類別 | baseline 32×32 | baseline 64×64 | base_loss (run 2) | class_weights (run 2) | two_stage Stage 2 |
|---|---|---|---|---|---|
| Center | 0.9128 | 0.9120 | 0.8883 | 0.7278 | 0.9580 |
| Donut | 0.8497 | 0.8187 | 0.7949 | 0.8427 | 0.8786 |
| Edge-Loc | 0.7750 | 0.8181 | 0.7479 | 0.6615 | 0.8930 |
| Edge-Ring | 0.9504 | 0.9763 | 0.9621 | 0.9408 | 0.9823 |
| Loc | 0.6652 | 0.7641 | 0.6181 | 0.5262 | 0.8323 |
| Near-full | 0.8980 | 0.9545 | 0.5641 | 0.9091 | 0.9302 |
| Random | 0.8246 | 0.8219 | 0.7037 | 0.7904 | 0.9157 |
| Scratch | 0.5371 | 0.7358 | 0.3077 | 0.2156 | **0.8701** |
| none | 0.9831 | 0.9892 | 0.9823 | 0.9478 | — |

---

## 輸出結果

每次執行後自動在 `outputs/` 下建立以實驗名稱與時間戳命名的目錄：

```
outputs/baseline_20260522_143012/
├── best_model.pth         # 驗證集 macro F1 最高的模型權重
├── metadata.json          # 模型設定、class_to_idx、best_val_f1 等資訊
├── training_curves.png    # Loss / F1 / Accuracy / Precision / Recall / LR 訓練曲線
└── confusion_matrix.png   # 行正規化的混淆矩陣（含原始計數）
```

兩階段訓練會分別在 `stage1/` 與 `stage2/` 子目錄各存一份。

---

## 後端 API

以 FastAPI 提供推論服務（`api/main.py`），模型固定使用 `outputs/two_stage_20260531_110545` 的兩階段權重，於服務啟動時載入。

### `GET /health`

**目的**：健康檢查，確認服務存活。

```json
{ "status": "ok" }
```

### `GET /wafers`

**目的**：回傳預先算好的晶圓樣本清單（`api/wafer.json`），供前端列表與統計圖表使用，包含整體良率與缺陷模式分佈。

```json
{
  "yield": 0.8621,                  // 所有樣本的平均良率（4 位小數）
  "total": 60,                      // 樣本總數
  "list": [
    {
      "id": 1,
      "height": 64,                 // 晶圓圖列數
      "width": 64,                  // 晶圓圖行數
      "wafer_map": [[0, 1, 2]],     // 二維陣列：0=晶圓外, 1=良品 die, 2=不良 die
      "die_total": 3059,            // 晶圓內 die 總數
      "defect_count": 313,          // 不良 die 數量
      "yield": 0.8977,              // 單片良率 =(die_total - defect_count) / die_total
      "pred_class": "none",         // 模型預測的缺陷模式名稱
      "pred_score": 0.9999          // 該預測類別的信心分數
    }
  ],
  "pattern_distribution": [         // 各缺陷模式的樣本數，固定 9 類皆會出現（無樣本則為 0）
    { "pred_class": "Center", "count": 5 }
  ],
  "pred_class": { "1": "Center", "9": "none" }  // 類別 id（LABEL_MAP index + 1）對照表
}
```

### `POST /explain`

**目的**：接收單張晶圓圖，執行兩階段推論並產生 Grad-CAM 熱力圖，說明模型判斷所依據的區域。

**輸入**：`{ "wafer": [[0, 1, 2, ...], ...] }` — 矩形二維陣列，元素僅允許 0/1/2，每邊長 1–2048，各列長度需一致。

```json
{
  "prediction": {
    "class_index": 7,               // LABEL_MAP index（0–8）
    "class_name": "Scratch",
    "confidence": 0.9312            // 預測類別的信心分數
  },
  "scores": [                       // 9 類分數，依 score 由大到小排序
    { "id": 8, "class": "Scratch", "score": 0.9312 }
  ],
  "explanation": {
    "heat_map": [[0.0, 0.35]],      // Grad-CAM 熱力值 0–1（2 位小數），已 resize 回原始尺寸
    "height": 64,
    "width": 64
  }
}
```

---

## 模組說明

### `data/dataset.py`

- **`extract_labeled_patterned_data(df)`** — 將原始 DataFrame 分為「有標記」、「有標記且有缺陷模式」、「有標記但無缺陷模式」三份。
- **`WaferDataset`** — 統一的 Dataset 類別，透過 `label_col` 與 `label_map` 參數支援單階段與兩階段的資料格式。內建 INTER_NEAREST resize 與 ImageNet 正規化。

### `models/builder.py`

- **`replace_head(model, num_classes)`** — 動態取代 EfficientNet 最後一層 Linear，適用於任意類別數。
- **`build_model(num_classes, device)`** — 載入 ImageNet 預訓練權重並替換 head，回傳已移至 device 的模型。

### `engine/trainer.py`

- **`run_one_epoch(...)`** — 單一 epoch 的訓練或評估，傳入 `optimizer=None` 即切換為 eval 模式。回傳 loss、acc、precision、recall、macro F1、per-class F1、confusion matrix。
- **`train_model(...)`** — 完整訓練迴圈，追蹤 best val F1 並儲存對應的 model state。

### `engine/visualize.py`

- **`plot_history(history)`** — 繪製訓練曲線（Loss、F1、Accuracy、Precision、Recall、LR）。
- **`plot_cm(cm, num_classes)`** — 繪製行正規化混淆矩陣，每格同時顯示比例與原始數量。
