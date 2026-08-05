# ADR-0002：推論路徑建模時不載入 ImageNet 預訓練權重

- 狀態：已採用（Accepted）
- 日期：2026-08-05
- 影響範圍：`models/builder.py`、`explain/core.py`、`api/explain_service.py`、容器啟動流程

---

## 背景

`build_model()` 原本無條件把 torchvision 的預訓練權重帶進 constructor：

```python
model_fn, weights = _MODEL_REGISTRY[model_name]   # (efficientnet_b4, EfficientNet_B4_Weights.IMAGENET1K_V1)
model = model_fn(weights=weights)
```

torchvision 看到 `weights=` 不是 `None`，就會去 `download.pytorch.org` 抓 checkpoint（EfficientNet-B4 的 `IMAGENET1K_V1` 約 75MB）存到 `~/.cache/torch/hub/checkpoints/`，然後把參數填進模型。

問題是**推論路徑根本不需要這份權重**。[explain/core.py:43-50](../../explain/core.py#L43-L50) 建完模型的下一件事就是：

```python
model = build_model(...)
state = torch.load(output_dir / "best_model.pth", map_location=device)
model.load_state_dict(state)
```

`load_state_dict()` 預設 `strict=True`，代表 `best_model.pth` 必須涵蓋模型的**每一個** parameter 與 buffer，少一個就會拋錯。換句話說：只要這行沒爆，ImageNet 權重的每一個數字都已經被自己訓練出來的權重整個蓋掉。那 75MB 下載對推論結果的貢獻是零。

訓練路徑不一樣：notebook 裡 `build_model(num_classes, device)` 是要拿 ImageNet 權重當 transfer learning 的起點，那份下載是有意義的。

### 為什麼在 Docker 裡這不只是「浪費一點時間」

三件事疊起來讓它變成實質的部署問題：

1. **容器啟動需要外網**。映像本身沒有這份 checkpoint，`docker run` 的當下才去打 `download.pytorch.org`。在無外網或有 egress 限制的環境，服務直接起不來。
2. **冷啟動變慢**。兩個 stage 的模型 = 兩次 `build_model()`，下載 75MB 才能開始服務（同一個 `model_name` 第二次會命中同進程的 cache，但第一次一定要等）。
3. **每次重啟都要重下**。`~/.cache/torch` 在容器的可寫層裡，[docker-compose.yml](../../docker-compose.yml) 沒有掛 volume 給它，容器一重建 cache 就沒了。

而且這發生的時機特別差：[api/explain_service.py:17-18](../../api/explain_service.py#L17-L18) 是在 **module import 時**就 `load_model()`，不是等第一個 request。所以下載卡在 uvicorn 起 app 的那一刻——健康檢查還沒開始回應、port 還沒 listen，外面看到的是「服務起不來」而不是「第一個 request 比較慢」。

## 決策

`build_model()` 保留 `pretrained` 參數，**預設 `True`**（給訓練用），推論路徑明確傳 `pretrained=False`：

```python
# models/builder.py
def build_model(num_classes, device, model_name=DEFAULT_MODEL_NAME, pretrained=True):
    model_fn, weights = _MODEL_REGISTRY[model_name]
    model = model_fn(weights=weights if pretrained else None)
    ...
```

```python
# explain/core.py — load_model()
model = build_model(
    num_classes=metadata["num_classes"],
    device=device,
    model_name=metadata["model_name"],
    pretrained=False,
)
```

`weights=None` 時 torchvision 只做隨機初始化，不碰網路、不碰 `~/.cache/torch`。後端映像因此**不再需要任何外網存取就能啟動**。

預設值選 `True` 而不是 `False`，是因為呼叫端數量懸殊：訓練 notebook 有 5 處（`train/*.ipynb`）都要預訓練權重，推論只有 `load_model()` 一處。讓少數的那一邊明講。

## 這樣做為什麼是安全的

擔心的點是「不載入預訓練權重，會不會有某些參數沒被 checkpoint 覆蓋到，變成隨機值？」——不會，而且有機制保證：

- `load_state_dict()` 預設 `strict=True`，任何 missing / unexpected key 都會直接拋 `RuntimeError`，不會靜默留下隨機參數。
- EfficientNet 的 BatchNorm `running_mean` / `running_var` 是 buffer，也在 `state_dict()` 裡，同樣受 strict 檢查保護。
- `replace_head()` 換掉的最後一層本來就不在 ImageNet 權重的形狀範圍內（1000 類 → 專案類別數），原本也是隨機初始化後靠 checkpoint 覆蓋。

也就是說，`pretrained=True` 與 `pretrained=False` 在 `load_state_dict()` 成功之後的模型狀態是**逐位元相同**的。

## 考慮過的其他方案

| 方案 | 為什麼不採用 |
|---|---|
| A. 掛 volume 給 `~/.cache/torch` | 只是把重下的頻率降低，第一次還是要外網；等於為一份不需要的資料維護一個 volume 與它的生命週期。 |
| B. **推論路徑傳 `pretrained=False`（採用）** | — |

## 後果

**正面**

- 後端容器啟動不需要外網，冷啟動少掉 75MB 下載。
- 容器重建不再有 cache 失效的懲罰——因為根本沒有 cache 要維護。
- `docker-compose.yml` 不需要為 torch cache 掛 volume。
- 推論映像與 `download.pytorch.org` 的可用性完全脫鉤。

**代價與已知待辦**

- `build_model()` 現在有兩種行為模式，呼叫端必須知道自己在哪一邊。誤傳 `pretrained=True` 的症狀是「能跑，只是慢」，不會有測試會抓到——這是這個設計唯一的軟肋。
- **模型仍在 import 時載入**（[api/explain_service.py:17-18](../../api/explain_service.py#L17-L18)）。本 ADR 只移除了下載，沒有改變載入時機；`torch.load()` + 兩個模型上 device 仍然阻塞 app 啟動。若之後要加 readiness probe 或縮短啟動時間，應改成 FastAPI 的 lifespan handler 或 lazy load，那是另一個決策。
- `best_model.pth` 與 `metadata.json` 目前是 `COPY . /app` 進映像的，映像大小仍受 checkpoint 影響；若要走「模型從外部掛載或下載」的路線，那也是獨立的決策。

