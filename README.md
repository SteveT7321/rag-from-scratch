# RAG 研究專案

> 從零手刻 RAG 系統，逐步探索各種進階技術。
> 示範文件：Subaru BRZ 2024 台灣市場車型冊（繁體中文 PDF）

---

## 研究路線圖

```
01_naive_rag          ← 從這裡開始（已完成）
    │
    ├─ 02_advanced_rag    升級取回精度（HyDE、Re-ranking、Query Rewriting）
    │
    ├─ 03_graph_rag       知識圖譜 + 多跳推理
    │
    ├─ 04_multimodal_rag  處理圖片和表格
    │
    ├─ 05_agentic_rag     LLM 自主決策取回策略
    │
    └─ 06_rag_fusion      多查詢 + RRF 融合
```

---

## 各方向一覽

| # | 方向 | 核心技術 | 解決的問題 | 難度 | 狀態 |
|---|------|---------|-----------|------|------|
| 01 | Naive RAG | Embedding + Cosine Similarity | 基礎 RAG 管線 | ★☆☆ | 完成 |
| 02 | Advanced RAG | HyDE、Cross-Encoder Re-ranking | 措辭不匹配、取回不相關 | ★★☆ | 待實作 |
| 03 | Graph RAG | 知識圖譜、BFS/DFS 搜尋 | 多實體關係推理 | ★★★ | 待實作 |
| 04 | Multimodal RAG | Vision LLM、表格結構化 | 圖片和表格資訊遺失 | ★★☆ | 待實作 |
| 05 | Agentic RAG | ReAct、Self-RAG、CRAG | 複雜多步問題、取回品質差 | ★★★ | 待實作 |
| 06 | RAG Fusion | 多查詢生成、RRF | 召回率低、視角單一 | ★★☆ | 待實作 |

---

## 共用資源

- `brz_tw.pdf` — 所有子專案的共同來源文件
- 每個 folder 各自維護自己的 `index/` 和 `requirements.txt`

---

## 技術棧總覽

```
全部專案共同：
  - Python 3.8+
  - Ollama（本地 LLM）
    ├── nomic-embed-text（Embedding，274MB）
    └── gemma4:e4b（生成 + Vision，9.6GB）

各專案額外：
  02: sentence-transformers（Cross-Encoder Re-ranking）
  04: Pillow（圖片處理）
  03/05/06: 純標準庫
```

---

## 快速開始

```bash
# 環境需求
ollama pull nomic-embed-text
ollama pull gemma4

# 從 01 開始
cd 01_naive_rag
python -m pip install pymupdf==1.23.8 numpy
python rag.py ingest ../brz_tw.pdf
python rag.py chat
```
