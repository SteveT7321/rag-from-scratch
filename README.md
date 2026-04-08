# RAG 研究專案

從零手刻六種 RAG 架構，不依賴 LangChain / LlamaIndex 等框架。
示範文件：Subaru BRZ 台灣市場車型冊（繁體中文 PDF，11 頁）。

---

## 技術棧

```
語言    Python 3.8+
推理    Ollama（本地，無需 API Key）
  ├── nomic-embed-text   Embedding，768 維，274 MB
  └── gemma4:e4b         文字生成 + Vision，9.6 GB
向量庫  numpy（純手刻，無 FAISS / ChromaDB）
PDF    PyMuPDF（pymupdf==1.23.8）
```

---

## 研究路線圖

六個資料夾各自獨立，但共用 `01_naive_rag/index/` 做為基礎向量索引：

```
01_naive_rag      ← 起點：最小可用 RAG
    │
    ├── 02_advanced_rag    提升取回精度（HyDE、Query Rewriting、Re-ranking）
    ├── 03_graph_rag       知識圖譜 + 多跳推理
    ├── 04_multimodal_rag  Vision LLM 解讀 PDF 圖表
    ├── 05_agentic_rag     ReAct 自主決策搜尋
    └── 06_rag_fusion      多查詢 + Reciprocal Rank Fusion
```

---

## 各方案一覽

| # | 方案 | 核心技術 | 解決的問題 |
|---|------|---------|-----------|
| 01 | [Naive RAG](01_naive_rag/) | Cosine Similarity Top-K | 建立基線，理解完整管線 |
| 02 | [Advanced RAG](02_advanced_rag/) | HyDE、Query Rewriting、LLM Re-ranking | 查詢措辭與文件不匹配 |
| 03 | [Graph RAG](03_graph_rag/) | 三元組抽取、BFS 圖遍歷 | 多實體關係、多跳推理 |
| 04 | [Multimodal RAG](04_multimodal_rag/) | Vision LLM 描述圖片 | PDF 圖表/配備表資訊遺失 |
| 05 | [Agentic RAG](05_agentic_rag/) | ReAct（思考→工具→觀察） | 複雜多步問題、動態搜尋 |
| 06 | [RAG Fusion](06_rag_fusion/) | 多子查詢 + RRF 排名融合 | 召回率低、單一視角查詢 |

---

## 核心概念速查

### Naive RAG 管線
```
PDF → PyMuPDF 文字提取 → Chunking（500字/50重疊）
    → nomic-embed-text → numpy 向量存儲
    → 查詢時餘弦相似度 Top-K → gemma4 生成回答
```

### HyDE（Hypothetical Document Embedding）
先讓 LLM 生成一個假設答案，用假設答案的向量去搜尋。
假設答案的措辭比問題更接近文件本身的表達方式。

### Query Rewriting
LLM 將一個問題改寫成 N 個不同角度的子查詢，擴大搜尋覆蓋面。

### LLM Re-ranking
向量搜尋取回更多候選（12個），再讓 LLM 逐一打分（0–10），
依分數重排後取 Top-K，精度高於純向量相似度。

### Graph RAG
從文件抽取 `(主體, 關係, 客體)` 三元組建圖，
查詢時先找種子節點，再 BFS 展開，擷取相關子圖送入 LLM。

### Multimodal RAG
用 PyMuPDF 提取 PDF 每頁圖片，送給 gemma4 Vision 生成文字描述，
描述文字再 embed 存入索引，讓圖表資訊也能被向量搜尋命中。

### Agentic RAG（ReAct）
LLM 自主決定每一步行動：
```
思考 → 工具（search / lookup / finish）→ 觀察結果 → 再思考
```
最多 6 步，比固定管線更靈活。

### RAG Fusion + RRF
生成 N 個子查詢 → 各自搜尋 → Reciprocal Rank Fusion 合併排名：
```
RRF(d) = Σ 1 / (60 + rank_i(d))
```

---

## 快速開始

```bash
# 前置：確認 Ollama 模型已下載
ollama pull nomic-embed-text
ollama pull gemma4:e4b

# 安裝依賴（所有方案共用）
python -m pip install pymupdf==1.23.8 numpy

# Step 1：建立向量索引（只需執行一次，02/05/06 會共用）
cd 01_naive_rag
python rag.py ingest ../brz_tw.pdf

# Step 2：跑任意方案
python rag.py chat -q "BRZ 的引擎馬力是多少？"

cd ../02_advanced_rag
python advanced_rag.py chat -q "BRZ 的引擎規格？" --hyde

cd ../05_agentic_rag
python agentic_rag.py chat -q "BRZ 跟 GR86 有什麼差異？" --verbose

cd ../06_rag_fusion
python rag_fusion.py chat -q "BRZ 的安全配備？" --show-queries
```

---

## 共用資源

| 資源 | 路徑 | 說明 |
|------|------|------|
| 來源文件 | `brz_tw.pdf` | 所有方案共用的 PDF |
| 基礎索引 | `01_naive_rag/index/` | 02、05、06 預設讀取此索引 |
| Graph 索引 | `03_graph_rag/index/` | 含知識圖譜 graph.json |
| 圖片索引 | `04_multimodal_rag/index/` | 含 Vision 描述的 chunks |

---

## 已知限制

1. **PDF 文字層薄**：車型冊大量使用圖表排版，文字層只有規格數字和少量段落。
   純文字 RAG 只能命中馬力、排氣量等明確數值，顏色、配備表等均失效。

2. **gemma4 Re-ranking 精度有限**：以 0–10 分評分中文 chunk 時，結果不穩定。
   生產環境建議換用 Cross-Encoder 模型（如 `BAAI/bge-reranker`）。

3. **圖形資訊需 04 Multimodal RAG**：EyeSight 功能說明、輪胎規格等在圖片中，
   只有 04 方案能正確取回這類資訊。
