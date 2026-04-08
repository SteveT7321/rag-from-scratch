# 01 — Naive RAG（基礎檢索增強生成）

最簡單的 RAG 實作：PDF 切塊 → 向量嵌入 → 餘弦相似度搜尋 → LLM 生成回答。
不使用任何框架，全部純手刻。

---

## 原理

```
PDF
 │
 ▼ PyMuPDF 逐頁提取文字
 │
 ▼ Chunking（段落切分 + 滑動視窗）
 │   每塊 ≤ 500 字元，重疊 50 字元
 │
 ▼ Ollama nomic-embed-text → 768 維向量
 │   存為 index/embeddings.npy + index/chunks.pkl
 │
 ▼ 查詢時：query → 向量 → 餘弦相似度 → top-4 chunks
 │
 ▼ Prompt 組裝 → Ollama gemma4:e4b → 串流輸出
```

### 餘弦相似度實作
```python
q = query_vec / (np.linalg.norm(query_vec) + 1e-10)   # L2 正規化
norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
scores = (embeddings / norms) @ q                       # dot product = cosine
top_idxs = np.argsort(scores)[::-1][:k]
```

---

## 環境設置

```bash
# 安裝依賴
pip install pymupdf==1.23.8 numpy>=1.24.0

# 確認 Ollama 模型
ollama list   # 需有 nomic-embed-text 和 gemma4:e4b
ollama pull nomic-embed-text
```

---

## 使用方式

```bash
# 1. 建立索引（只需執行一次）
python rag.py ingest brz_tw.pdf

# 2. 單次查詢
python rag.py chat --query "BRZ 的引擎馬力是多少？"

# 3. 互動模式（輸入 exit 離開）
python rag.py chat
```

---

## 實測結果

來源文件：Subaru BRZ 台灣市場車型冊（繁體中文 PDF，11 頁）
索引大小：25 chunks

| 查詢 | 結果 | 說明 |
|------|------|------|
| BRZ 的引擎馬力是多少？ | ✅ **234 ps/rpm（EEC 淨）** | 文字層有明確數值 |
| BRZ 的排氣量是多少 cc？ | ✅ **2,387 cc** | 規格表文字可檢索 |
| 有哪些外觀顏色可以選擇？ | ❌ 資料中沒有相關資訊 | 顏色資訊在圖片/表格中 |
| EyeSight 安全系統有什麼功能？ | ❌ 資料中沒有相關資訊 | 功能說明為影像排版 |
| 車重是多少？ | ❌ 資料中沒有相關資訊 | 規格數字嵌在圖表中 |
| 輪胎規格？ | ❌ 資料中沒有相關資訊 | 配備表為圖片格式 |

---

## 優缺點分析

### 優點
- 架構簡單，易於理解與調試
- 對純文字型問題（數值、段落描述）命中率高
- 無外部依賴，部署成本低

### 缺點
- **對圖表、配備表、影像型資訊完全無效**（PDF 的文字層不含這些內容）
- 查詢措辭需與文件措辭接近，否則向量距離偏大
- Top-K 固定，沒有排序或篩選機制
- 無法處理多跳問題（需要多個 chunk 組合才能回答）

---

## 關鍵設定

| 參數 | 值 | 說明 |
|------|----|------|
| CHUNK_SIZE | 500 字元 | 段落超過此長度才切分 |
| CHUNK_OVERLAP | 50 字元 | 滑動視窗重疊，避免邊界截斷資訊 |
| TOP_K | 4 | 送入 LLM 的 chunk 數量 |
| EMBED_MODEL | nomic-embed-text | 768 維，支援多語言 |
| GEN_MODEL | gemma4:e4b | 本地 Ollama 推理 |

---

## 索引結構

```
index/
├── chunks.pkl      # List[Dict]，每筆含 chunk_id、page、text
└── embeddings.npy  # shape (25, 768)，float32
```

此索引被 02～06 所有進階方法共用（路徑：`../01_naive_rag/index/`）。

---

## 進階版本

| 方案 | 解決的問題 |
|------|-----------|
| [02 Advanced RAG](../02_advanced_rag/) | HyDE、Query Rewriting、Re-ranking 提升召回率 |
| [03 Graph RAG](../03_graph_rag/) | 知識圖譜，處理實體關係型問題 |
| [04 Multimodal RAG](../04_multimodal_rag/) | Vision LLM 解讀圖表，解決影像盲點 |
| [05 Agentic RAG](../05_agentic_rag/) | ReAct 自主決策搜尋策略 |
| [06 RAG Fusion](../06_rag_fusion/) | 多查詢 + RRF 融合提升召回率 |
