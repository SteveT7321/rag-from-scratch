# 06 — RAG Fusion

## 狀態：待實作

## 核心問題

單一查詢有**視角盲點**：使用者的問法不一定是最好的搜尋方式，而且不同查詢會找到不同但都相關的 chunk。

## 核心思路

**同一個問題，從多個角度問，再把結果融合**。

```
使用者問題："BRZ 值得買嗎？"
    │
    ▼
[LLM 生成 N 個子查詢]
  ├── "BRZ 性能規格和競爭對手比較"
  ├── "BRZ 安全配備"
  ├── "BRZ 駕駛樂趣和操控感"
  └── "BRZ 售價與配備"
    │
    ▼
[分別搜尋] × 4
  ├── 結果集 A（來自子查詢1）
  ├── 結果集 B（來自子查詢2）
  ├── 結果集 C（來自子查詢3）
  └── 結果集 D（來自子查詢4）
    │
    ▼
[Reciprocal Rank Fusion] 合併並重新排序
    │
    ▼
最終 Top-K chunks（比任何單一查詢都更全面）
    │
    ▼
LLM 生成最終答案
```

## Reciprocal Rank Fusion（RRF）

RRF 是融合多個排序列表的標準演算法，1973年由 Borda 提出的概念，RRF 是現代版本（Cormack et al., 2009）。

### 公式

每個 chunk 的 RRF 分數：

```
RRF_score(d) = Σ  1 / (k + rank_i(d))
               i

其中：
  d = 某個 chunk
  rank_i(d) = 在第 i 個查詢結果中的排名（從1開始）
  k = 常數，通常設 60（防止排名第1的權重過高）
```

### 範例計算

假設 "BRZ EyeSight 安全" 這個 chunk：
- 在子查詢1的結果中排名第 2
- 在子查詢3的結果中排名第 1
- 在子查詢4的結果中排名第 5

```
RRF = 1/(60+2) + 1/(60+1) + 1/(60+5)
    = 0.0161 + 0.0164 + 0.0154
    = 0.0479
```

### 為什麼 RRF 優於直接平均分數？

- 不同查詢的 cosine similarity 分數**不可直接比較**（不同基準）
- RRF 只看排名，跨查詢可比
- 出現在多個查詢結果裡的 chunk 自然得到更高分
- 對排名靠後的文件懲罰小（因為 k=60 的緩衝）

### 純手刻實作

```python
def reciprocal_rank_fusion(result_lists: List[List[Dict]], k: int = 60) -> List[Dict]:
    scores = {}  # chunk_id -> rrf_score
    
    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            cid = chunk["chunk_id"]
            scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank)
    
    # 依分數排序，取 Top-K
    sorted_ids = sorted(scores, key=scores.get, reverse=True)
    ...
```

## 子查詢生成策略

### 策略一：LLM 生成（語意多樣）

```python
prompt = f"""
針對以下問題，生成 4 個不同角度的搜尋查詢，用來在 Subaru BRZ 車型冊裡搜尋。
不同查詢應覆蓋不同面向。每行一個查詢，不要編號。

原始問題：{question}
"""
```

### 策略二：Query Decomposition（分解子問題）

把複合問題拆解成獨立的子問題：

「比較 BRZ MT 和 AT 版本的油耗和性能差異」
→ 「BRZ MT 版本油耗」
→ 「BRZ AT 版本油耗」
→ 「BRZ MT 版本加速性能」
→ 「BRZ AT 版本加速性能」

### 策略三：假設答案展開（結合 HyDE）

對每個子查詢各生成一個 HyDE 假設答案，再搜尋。

## 實作計畫

### 檔案結構

```
06_rag_fusion/
├── plan.md
├── requirements.txt
├── query_gen.py      # LLM 生成子查詢
├── rrf.py            # Reciprocal Rank Fusion
├── rag_fusion.py     # 整合 CLI
└── index/            # 共用 01_naive_rag 的索引，或重建
```

### CLI 設計

```bash
# 基本用法（LLM 自動生成 4 個子查詢）
python rag_fusion.py chat --query "BRZ 值得買嗎？"

# 指定子查詢數量
python rag_fusion.py chat --query "BRZ 安全性" --n-queries 6

# 顯示子查詢（除錯用）
python rag_fusion.py chat --query "..." --show-queries
```

**顯示子查詢的輸出範例：**

```
生成子查詢...
  1. BRZ 安全配備列表
  2. EyeSight 駕駛輔助功能說明
  3. BRZ 碰撞防護系統
  4. BRZ 被動安全 SRS 氣囊

分別搜尋並融合...
RRF Top-4 chunks：第 5, 5, 2, 8 頁（分數：0.082, 0.071, 0.063, 0.051）

BRZ 專家：BRZ 的安全配備包含...
```

## 預期效果

| 問題類型 | Naive RAG | RAG Fusion |
|----------|-----------|-----------|
| 開放性問題 | 只找到一個角度 | 多角度覆蓋 |
| 使用者措辭模糊 | 可能找不到 | 子查詢補救 |
| 需要比較的問題 | 只找到其中一邊 | 兩邊都找到 |
| 召回率（Recall） | 較低 | 顯著提升 |

**代價**：N 個子查詢 = N 倍的 embedding 呼叫次數

## 參考

- RAG-Fusion 論文：Forget RAG, the Future is RAG-Fusion (Raudaschl, 2023)
- RRF 論文：Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods (Cormack et al., 2009)
