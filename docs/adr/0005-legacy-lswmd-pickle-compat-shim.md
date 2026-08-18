# ADR-0005：以相容層讀取舊格式 LSWMD.pkl，取代不存在的 LSWMD_v2.pkl

- 狀態：已採用（Accepted）
- 日期：2026-08-18
- 影響範圍：`data/dataset.py`

---

## 背景

`data/dataset.py` 的 `load_raw_data()` 原本這樣寫：

```python
def load_raw_data():
    path = kagglehub.dataset_download("qingyi/wm811k-wafer-map")
    df = pd.read_pickle(f"{path}/LSWMD_v2.pkl")
    ...
```

在一台全新環境（kagglehub 快取為空）跑 `train/simple_train.ipynb` 時，這行直接炸掉：

```
No such file or directory: '...\wm811k-wafer-map\versions\1/LSWMD_v2.pkl'
```

### 排查：LSWMD_v2.pkl 從未存在於這個 Kaggle 資料集

用 Kaggle 公開 API（`GET /api/v1/datasets/view/qingyi/wm811k-wafer-map`）查證：

- 這個資料集自 2018-02-26 上傳後**只有一個版本**（`currentVersionNumber: 1`，`versionNotes: "Initial release"`），從未更新過。
- `totalBytesNullable: 2095505977`，與 kagglehub 實際下載到的 `LSWMD.pkl` 檔案大小**完全一致**——代表這個資料集本來就只包著這一個檔案。

另外搜尋 Kaggle 上所有含 `LSWMD_v2`、`wafer map v2`、`WM-811K v2` 等關鍵字的資料集，沒有任何一筆是官方或第三方提供、檔名為 `LSWMD_v2.pkl` 且結構相容的版本。

結論：`LSWMD_v2.pkl` 不是 Kaggle 上的檔案，而是**先前有人手動把 `LSWMD.pkl` 轉換成新版 pandas 相容格式後，另存的本機衍生檔**，存放在 kagglehub 的快取目錄裡。README 裡各實驗（`baseline`、`class_weights`、`two_stage` 等）能成功執行，靠的正是那份本機轉換過的快取檔案；一旦快取被清空或換一台機器，這個轉換步驟本身沒有被記錄在程式碼裡，就會直接失敗。

### 為什麼 LSWMD.pkl 讀不動

`LSWMD.pkl` 是用 pandas 0.x 年代（Python 2）序列化的。對整份 pickle bytecode 掃描 `GLOBAL` opcode，命中的 pandas/numpy 符號如下：

| 舊路徑 | 現況 |
| --- | --- |
| `pandas.core.frame.DataFrame` | 仍在原路徑，可正常解析 |
| `pandas.core.internals.BlockManager` | 仍在原路徑，可正常解析 |
| `pandas.indexes.base.Index` / `_new_Index` | **模組已搬到** `pandas.core.indexes.base` |
| `pandas.indexes.range.RangeIndex` | **模組已搬到** `pandas.core.indexes.range` |

直接 `pd.read_pickle()` 會在 `pandas.indexes.base` 這一步就以 `ModuleNotFoundError` 中止。就算繞過模組路徑問題，這份 pickle 是 Python 2 寫的（欄位值用 `str`，pickle 內部以 bytes 存放），Python 3 用預設的 `ascii` 編碼反序列化會再炸一次 `UnicodeDecodeError`；`pandas.read_pickle` 目前的版本不再提供 `encoding` 參數可調，因此也無法用它繞過。

### 驗證格式相容

用相容層（見「決策」）把 `LSWMD.pkl` 讀出來後，實際檢查了 schema 與資料內容：

- 欄位：`waferMap, dieSize, lotName, waferIndex, trianTestLabel, failureType`，共 811,457 列，和 WM-811K 論文描述的規模一致。
- `failureType` 的儲存形式和現有程式碼（`load_raw_data` 第 18–20 行的 `x[0][0]` 解包、`extract_labeled_patterned_data` 的 `_unwrap_failure_type`）預期的**完全一致**：已標記列是 `array([['Center']], dtype='<U6')` 這種巢狀陣列，未標記列是空陣列 `array([], shape=(0, 0))`。
- `trianTestLabel` 同樣是巢狀字串陣列，未標記列一樣是空陣列（638,507 列無標記、118,595 列 Test、54,355 列 Training，與已知的 WM-811K 統計數字吻合）。

也就是說，**原始 `LSWMD.pkl` 的資料 schema 本來就和專案程式碼預期的一致**，唯一的障礙是 pandas 版本造成的反序列化失敗，不是資料格式（v1/v2）本身的差異。

## 決策

在 `data/dataset.py` 加入 `_load_legacy_lswmd_pickle()`，只在 `LSWMD_v2.pkl` 快取不存在時才啟用：

```python
def _load_legacy_lswmd_pickle(pkl_path):
    import pandas.core.indexes.base as _idx_base
    import pandas.core.indexes.range as _idx_range

    added = {
        "pandas.indexes": _idx_base,
        "pandas.indexes.base": _idx_base,
        "pandas.indexes.range": _idx_range,
    }
    previous = {name: sys.modules.get(name) for name in added}
    sys.modules.update(added)
    try:
        with open(pkl_path, "rb") as f:
            return pickle.load(f, encoding="latin1")
    finally:
        for name, mod in previous.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


def load_raw_data():
    path = kagglehub.dataset_download("qingyi/wm811k-wafer-map")
    v2_path = f"{path}/LSWMD_v2.pkl"

    if os.path.exists(v2_path):
        df = pd.read_pickle(v2_path)
    else:
        df = _load_legacy_lswmd_pickle(f"{path}/LSWMD.pkl")
        df.to_pickle(v2_path)

    df["failureType"] = df["failureType"].apply(
        lambda x: x[0][0] if isinstance(x, np.ndarray) and x.size > 0 else x
    )
    return df
```

重點設計：

1. **模組映射用 `sys.modules`，且執行完還原**——只在 `pickle.load()` 這一次呼叫期間，把 `pandas.indexes*` 指向現有的 `pandas.core.indexes.*` 模組物件，讀取結束（含例外）就把 `sys.modules` 還原成呼叫前的狀態，避免污染其他程式碼對 `pandas.indexes` 這個路徑的（不存在的）預期。
2. **`encoding="latin1"`**——繞開 Python 2 pickle 對 3 的 `UnicodeDecodeError`。這是讀取跨大版本 pickle 的標準做法（`numpy`/`sklearn` 生態中常見），latin1 是雙向映射（每個 byte 對應一個 code point），不會遺失資料。
3. **轉出的結果快取回 `LSWMD_v2.pkl`**——相容層每次讀取整份 2GB pickle 需要一定時間，轉換一次後用 `to_pickle` 存回 kagglehub 快取目錄，下次啟動直接命中 `pd.read_pickle(v2_path)` 的快速路徑，成本只在第一次真正發生。這同時也讓 `LSWMD_v2.pkl` 這個檔名「名符其實」：它就是本機產生的、可以直接被現有 pandas 讀取的版本。

## 驗證

在乾淨的 kagglehub 快取（僅有原始 `LSWMD.pkl`）下執行 `load_raw_data()`：

- 相容層成功解析 811,457 列，欄位與 dtype 符合預期。
- `df.to_pickle(v2_path)` 寫出約 2GB 的 `LSWMD_v2.pkl`。
- 之後續跑改走 `pd.read_pickle(v2_path)`，回傳的 `failureType` 分佈與轉換前一致。

## 後果

**正面**

- 全新環境（新機器、CI、清空快取後）可以直接從 kagglehub 公開下載的 `LSWMD.pkl` 重建出可用資料，不再依賴「某人手上剛好有轉換好的 `LSWMD_v2.pkl`」這種未被程式碼記錄的隱性前提。
- 轉換只在第一次發生，後續行為與原本讀 `LSWMD_v2.pkl` 完全相同，不影響既有 notebook 的執行時間。

**代價與注意事項**

- 首次執行仍需要读入並反序列化整份 ~2GB 的舊格式 pickle，比直接讀 v2 慢，且需要一次性的額外磁碟空間（`LSWMD.pkl` + 新寫出的 `LSWMD_v2.pkl` 同時存在，合計約 4GB）。
- 相容層目前只映射了這份 pickle 實際用到的兩個舊模組路徑（`pandas.indexes.base`、`pandas.indexes.range`）。如果未來 Kaggle 上出現內容不同、序列化時用到其他舊路徑（例如舊版 `Block` 子類別）的檔案，仍會需要擴充映射表。
- 這個相容層是讀取 pandas 0.x 年代 pickle 的通用手法，不是 WM-811K 專屬技巧；一旦某天資料集擁有者補上真正的 `LSWMD_v2.pkl`，`load_raw_data()` 會直接透過 `os.path.exists` 檢查優先使用它，相容層自動不會被觸發。

## 曾考慮的其他方案

**在 Kaggle 上尋找提供 `LSWMD_v2.pkl` 的替代資料集**
實際搜尋（Kaggle 公開 API 關鍵字搜尋 + 逐一檢查候選資料集的 metadata/description）沒有找到任何檔名或內容相符的資料集。找到的幾個「WM-811K 相關」資料集（`mohammedfariskhan/wm811k-clean-subset`、`nithinbandi/wm-811k-augmented-and-preprossed` 等）都是不同的前處理/切分方式，換過去會需要重寫 `load_raw_data()` 下游的解析邏輯，且資料切分方式與 README 中既有實驗結果的可比性會被破壞。放棄此方案。
