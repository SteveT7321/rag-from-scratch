# 01 — Naive RAG

## 狀態：已完成

## 核心概念

最基礎的 RAG 實作，展示最小可用系統的全部環節。

```
PDF → 文字提取 → 固定大小 Chunking → Embedding → 向量存儲
問題 → Embedding → Cosine Similarity → Top-K → LLM → 回答
```

## 關鍵設計決策

| 項目 | 選擇 | 原因 |
|------|------|------|
| Chunking | 段落切 + 固定大小 | 最簡單，兼顧語意邊界 |
| 向量存儲 | numpy .npy | 零依賴，25 chunks 夠用 |
| 相似度 | Cosine Similarity | 標準做法，數值穩定 |
| Embedding | nomic-embed-text | 輕量、多語言、免費 |
| LLM | gemma4:e4b | 本地，無需 API key |

## 已知問題（留給後續 folder 解決）

1. **單一查詢問題**：使用者的問題措辭若和文件不同，容易找不到
   → 解法見 `02_advanced_rag`（Query Rewriting）、`06_rag_fusion`（多查詢）

2. **Chunk 之間無關聯**：只知道每個片段的內容，不知道實體之間的關係
   → 解法見 `03_graph_rag`

3. **圖片、表格無法處理**：PDF 裡的圖表被丟掉
   → 解法見 `04_multimodal_rag`

4. **固定取 Top-K，不管相關性好不好**：有時候取回的 chunk 根本不相關
   → 解法見 `02_advanced_rag`（Re-ranking）、`05_agentic_rag`（自評估）

## 檔案

- `rag.py` — 主程式
- `requirements.txt` — pymupdf, numpy
- `index/` — 已建立的向量索引（25 chunks × 768 dim）

## 執行

```bash
python rag.py ingest ../brz_tw.pdf
python rag.py chat
python rag.py chat --query "BRZ 的引擎規格？"
```
