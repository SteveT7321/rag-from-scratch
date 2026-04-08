# 04 — Multimodal RAG

用 Vision LLM（gemma4:e4b）為 PDF 圖片生成文字描述，再 embed 存入索引。
文字 chunk 和圖片描述 chunk 混合在同一個向量空間中，統一搜尋。
解決 Naive RAG 對圖片、表格、車色圖等「視覺資訊」完全盲目的問題。

---

## 原理

### ingest 流程

```
PDF
 │
 ├─ 文字層 → PyMuPDF 提取 → Chunking → 25 個文字 chunks
 │
 └─ 圖片層 → PyMuPDF get_images() → 過濾小圖（< 80×80px）
              ↓
         縮圖至 1024px → base64 → Ollama Vision API
              ↓
         gemma4 生成繁體中文描述（顏色、配件、文字、用途）
              ↓
         描述文字 = 圖片 chunk（可被 embed 和搜尋）

所有 chunks（文字 + 圖片描述）統一 embed → embeddings.npy
```

### 查詢流程

```
問題 → embed → 餘弦相似度搜尋 → Top-4 chunks
                                  ↓
                    （可能包含文字 chunk 或圖片描述 chunk）
                                  ↓
                         LLM 生成回答
```

### 關鍵設計

- 圖片描述 chunk 的 `type` 欄位標記為 `"image"`，方便追蹤
- 縮圖上限 1024px，控制 Vision LLM 的 token 消耗
- CMYK / 非 RGB 色域自動轉換（PyMuPDF 相容性修正）

---

## 使用方式

```bash
# 1. 建立索引（需時約 20 分鐘：56 張圖 × Vision LLM）
python multimodal_rag.py ingest ../brz_tw.pdf

# 2. 查詢
python multimodal_rag.py chat -q "BRZ 的車身是什麼顏色？"

# 3. 查看提取的圖片清單與描述
python multimodal_rag.py list-images

# 4. 互動模式
python multimodal_rag.py chat
```

---

## 實測結果

### 索引統計

| 項目 | 數值 |
|------|------|
| 文字 chunks | 25 個 |
| 圖片 chunks | 56 個（過濾 < 80×80px 後） |
| 總 chunks | 81 個 |
| 索引維度 | (81, 768) |
| ingest 時間 | 約 20 分鐘 |

---

### 查詢：「BRZ 的車身是什麼顏色？」

參考頁碼：第 4, 6 頁（2 文字 + 2 圖片描述）

✅ 回答：
> 在不同圖片中展現了多種顏色：
> 1. **淺灰色/銀灰色（Silver/Light Gray）** — 第6頁運動渲染圖
> 2. **白色（White）** — 第4頁展示圖

> Naive RAG 對此查詢完全失敗（顏色資訊在圖片中），Multimodal RAG 透過 Vision 描述成功識別。

---

### 查詢：「BRZ 的規格數值，馬力扭力等？」

參考頁碼：第 2, 3, 10 頁（4 文字 + 0 圖片）

✅ 完整規格回答：
```
引擎：SUBARU BOXER 水平對臥四缸自然進氣 2.4L
排氣量：2,387 cc
壓縮比：12.5:1
最大馬力：234 ps / 7,000 rpm（EEC淨）
最大扭力：25.5 kgfm / 3,700 rpm（EEC淨）
極速：215 km/h
0-100 km/h：6.9 秒
全長：4,265 mm / 全寬：1,775 mm / 全高：1,310 mm
軸距：2,575 mm / 車重：1,310 kg
```

---

### 查詢：「BRZ 有哪些車身顏色可以選擇？」

❌ 資料中沒有相關資訊

原因：車型冊的顏色選項頁面為圖表格式，圖片中只展示個別顏色樣本，
Vision LLM 無法從單張圖片推斷「完整顏色選項清單」。
查詢措辭需改為「BRZ 的車身是什麼顏色」才能命中個別圖片描述。

---

### list-images 樣本

```
第1頁 | 3745×2498px
  描述：山區拍攝的動態廣告照，Subaru BRZ 運動型轎跑車...

第2頁 | 1482×1482px
  描述：深藍色金屬光澤（Metallic Blue）BRZ，蜿蜒道路動感照...

第3頁 | 2318×1482px
  描述：深寶石藍色 BRZ，車身線條流暢，極具運動感...
```

---

## 效果分析

### 優點
- **突破文字層限制**：顏色、外觀細節等純視覺資訊可被搜尋
- **文字+圖片統一索引**：同一個向量空間，不需要雙軌搜尋
- **描述可讀性高**：Vision LLM 生成的描述具體（顏色名稱、尺寸、用途）

### 缺點
- **ingest 時間極長**：56 張圖 × Vision LLM ≈ 20 分鐘（是純文字的 10 倍）
- **描述精度依賴模型**：gemma4 對細小圖片（418×259px）的描述準確度下降
- **無法從圖推論清單**：「有哪些顏色？」需要圖片本身顯示完整選項才能回答
- **圖片 chunk 有雜訊**：背景風景、裝飾性小圖也會被描述並 embed，稀釋相關性

### 與 Naive RAG 對比

| 查詢 | Naive RAG | Multimodal RAG |
|------|-----------|---------------|
| 引擎馬力 | ✅ 234 ps | ✅ 234 ps + 完整規格 |
| 車身顏色（描述型） | ❌ | ✅ 銀灰色、白色 |
| 完整顏色選項清單 | ❌ | ❌（需圖片含完整列表） |
| 車體尺寸 | ❌ | ✅ 全長/全寬/全高/軸距/車重 |

---

## 關鍵設定

| 參數 | 值 | 說明 |
|------|----|------|
| MIN_IMG_W / MIN_IMG_H | 80px | 過濾 icon、裝飾性小圖 |
| MAX_IMG_DIM | 1024px | 縮圖上限，控制 Vision token |
| TOP_K | 4 | 搜尋回傳 chunk 數 |
| GEN_MODEL | gemma4:e4b | 文字生成 + Vision 描述共用 |

---

## 索引結構

```
index/
├── chunks.pkl         # 81 個 chunks（25 文字 + 56 圖片）
├── embeddings.npy     # shape (81, 768)
└── img_metadata.json  # 56 張圖片的路徑、尺寸、完整描述

images/
├── page01_img00.png
├── page02_img01.png
└── ...（56 張，.gitignore 已排除）
```
