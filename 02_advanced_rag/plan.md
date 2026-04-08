# 02 — Advanced RAG

## 狀態：待實作

## 核心問題

Naive RAG 最大的硬傷：**使用者的問題和文件的寫法不一樣**。

- 使用者問：「BRZ 跑起來快嗎？」
- 文件寫的：「最大馬力 234 ps，0-100 km/h 加速 6.3 秒」

兩句話語意相近，但 embedding 的向量距離未必夠近。

## 三個升級方向

### 1. HyDE（Hypothetical Document Embeddings）

**原理**：先讓 LLM 捏造一個「假設性答案」，再用這個假設答案去搜向量庫，而不是直接用問題搜。

為什麼有效？因為假設答案的措辭風格會更接近文件本身的寫法。

```
使用者問題
    │
    ▼
LLM 生成假設性答案（不看任何文件，純猜）
    │  "BRZ 應該有 2.0L 引擎，馬力大概 200 hp..."
    ▼
對假設答案做 Embedding
    │
    ▼
向量搜尋（比原問題找得更準）
    │
    ▼
用真實取回的 chunks + 原問題 → LLM 生成真正回答
```

**代價**：多一次 LLM 呼叫（用小模型即可）

---

### 2. Query Rewriting / Expansion（查詢改寫）

**原理**：把一個問題改寫成多個角度，覆蓋更多可能的措辭。

```
原問題："BRZ 安全嗎？"
    │
    ▼
LLM 改寫成：
  - "BRZ 有哪些安全配備？"
  - "BRZ 的碰撞防護系統"
  - "EyeSight 駕駛輔助功能"
    │
    ▼
三個問題分別搜尋，合併結果去重
```

---

### 3. Re-ranking（重新排序）

**原理**：Embedding 搜尋速度快但精度粗；先用 Embedding 取出 Top-20，再用 Cross-Encoder 精排前 4。

```
問題 + 文件片段 → Cross-Encoder → 精確相關性分數
```

Cross-Encoder 不像 Bi-Encoder 各自獨立 encode，而是把問題和文件一起輸入，能看到兩者的交互關係，精度遠高於 cosine similarity。

```
Embedding 搜尋（快，粗）: 取 Top-20 candidates
    │
    ▼
Cross-Encoder 重排（慢，精）: 從 20 個裡精選 Top-4
    │
    ▼
LLM 生成
```

**工具**：`sentence-transformers` 的 CrossEncoder，或 Ollama 的 reranker 模型

---

## 實作計畫

### 檔案結構

```
02_advanced_rag/
├── plan.md
├── requirements.txt
├── hyde.py          # HyDE 實作
├── rewrite.py       # Query Rewriting 實作
├── rerank.py        # Cross-Encoder Re-ranking 實作
└── advanced_rag.py  # 整合版 CLI（可開關各功能）
```

### 建議套件

```
pymupdf==1.23.8
numpy
sentence-transformers   # Cross-Encoder re-ranking
requests
```

### CLI 設計

```bash
# 開啟 HyDE
python advanced_rag.py chat --hyde --query "BRZ 跑起來快嗎？"

# 開啟 Query Rewriting
python advanced_rag.py chat --rewrite --query "BRZ 安全性如何？"

# 開啟 Re-ranking
python advanced_rag.py chat --rerank --query "引擎規格"

# 全開
python advanced_rag.py chat --hyde --rewrite --rerank -q "BRZ 值得買嗎？"
```

## 預期改善

| 場景 | Naive RAG | Advanced RAG |
|------|-----------|-------------|
| 問法和文件措辭不同 | 容易找不到 | HyDE / Rewrite 補救 |
| 取回的 chunk 不相關 | 照單全收 | Re-ranking 過濾 |
| 複雜問題需多角度 | 單一查詢 | Rewrite 展開多面向 |

## 參考

- HyDE 論文：Precise Zero-Shot Dense Retrieval without Relevance Labels (Gao et al., 2022)
- Re-ranking：Cross-Encoders vs Bi-Encoders (SBERT 文件)
