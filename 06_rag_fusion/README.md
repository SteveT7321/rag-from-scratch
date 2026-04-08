# 06 — RAG Fusion + Reciprocal Rank Fusion

同一個問題從多角度生成子查詢，分別向量搜尋，再用 RRF 融合排名。
比單一查詢召回率更高，對開放性問題效果顯著。
索引共用 `01_naive_rag/index/`，不需要重新 ingest。

---

## 原理

### 流程

```
原始問題
 │
 ▼ LLM 生成 N 個子查詢（預設 N=4）
 │
 ▼ 每個子查詢 → embed → 向量搜尋 Top-6
 │
 ▼ Reciprocal Rank Fusion 融合 N 個排名列表
 │
 ▼ RRF Top-5 chunks → LLM 生成回答
```

### RRF 公式

```
RRF(d) = Σ_i  1 / (k + rank_i(d))    k = 60
```

- `rank_i(d)`：文件 d 在第 i 個查詢結果中的排名（從 1 開始）
- `k=60`：平滑常數，減少頂排名文件的過度優勢
- 出現在多個查詢結果中的文件會累積更高分數

**核心思想**：一個 chunk 若被多個不同角度的查詢都找到，代表它真正相關。

---

## 使用方式

```bash
# 基本查詢（預設生成 4 個子查詢）
python rag_fusion.py chat -q "問題"

# 顯示子查詢和 RRF 分數
python rag_fusion.py chat -q "問題" --show-queries

# 指定子查詢數量
python rag_fusion.py chat -q "問題" -n 6

# 互動模式
python rag_fusion.py chat --show-queries

# 若需要自建索引（預設共用 01 的索引）
python rag_fusion.py ingest ../brz_tw.pdf
```

---

## 實測結果

索引：25 chunks（共用 01_naive_rag/index/）

### 查詢：「BRZ 的引擎馬力是多少？」（--show-queries）

生成 5 個子查詢（原始 + 4 個）：
```
（原始）  BRZ 的引擎馬力是多少？
（子查詢1）引擎馬力
（子查詢2）動力輸出參數
（子查詢3）BRZ 規格 馬力
（子查詢4）台灣市售 BRZ 動力數據
```

RRF Top-5 結果：

| Chunk | RRF 分數 | 頁碼 |
|-------|---------|------|
| chunk0 | 0.0820 | 第 2 頁 |
| chunk15 | 0.0796 | 第 3 頁 |
| chunk1 | 0.0787 | 第 6 頁 |
| chunk17 | 0.0786 | 第 10 頁 |
| chunk6 | 0.0611 | 第 2 頁 |

回答：✅ **BRZ 的引擎馬力是 234 ps（172 kW）**

---

### 查詢：「BRZ 有哪些安全配備？」（--show-queries）

生成 5 個子查詢：
```
（原始）  BRZ 有哪些安全配備？
（子查詢1）新車 BRZ 安全配置等級
（子查詢2）BRZ 最新款行車安全科技有哪些
（子查詢3）Subaru BRZ 台灣版車型 安全防護系統檢視
（子查詢4）BRZ 安全配備是否包含車主適應巡航等高階功能
```

RRF Top-5：第 2, 7, 10 頁

回答：✅ 詳細列出 SRS 氣囊系統（雙前座、側邊、車側簾式、膝部）、ELR 預縮式安全帶等配備。

> 對比 Naive RAG 查詢「EyeSight 安全系統」完全失敗，RAG Fusion 用更廣角的子查詢（「安全配置等級」「行車安全科技」）成功命中包含氣囊/安全帶資訊的 chunk。

---

## 效果分析

### 優點
- **更高召回率**：不同角度的子查詢覆蓋更多措辭和面向，降低 embedding 距離偏差的影響
- **RRF 合理融合**：不依賴絕對分數，只用相對排名，各查詢間可比性強
- **穩定性高**：單一子查詢差也沒關係，RRF 會被其他查詢的信號補償
- **實作簡單**：相比 Agentic RAG，流程固定、延遲可預測

### 缺點
- **LLM 呼叫次數多**：生成子查詢 1 次 + N 次 embed，整體延遲高於單次查詢
- **子查詢品質依賴 LLM**：若 LLM 生成的子查詢角度重疊，效果遞減
- **不適合「一定要在第 X 頁」的精準查詢**：仍是 Top-K 近似，非精確查找

---

## 與 Naive RAG 對比

| 問題 | Naive RAG | RAG Fusion |
|------|-----------|------------|
| 引擎馬力 | ✅ 234 ps | ✅ 234 ps（更多佐證頁） |
| 安全配備 | ❌ 資料中沒有相關資訊 | ✅ 詳細氣囊/安全帶清單 |
| 顏色選擇 | ❌ | ❌（仍在圖片層）|

RAG Fusion 對「安全配備」這類廣義問題效果顯著提升，
但對圖片層資訊（顏色、規格圖表）仍無法處理。

---

## 關鍵設定

| 參數 | 值 | 說明 |
|------|----|------|
| DEFAULT_N | 4 | 預設子查詢數量 |
| TOP_K_PER_Q | 6 | 每個子查詢取幾個候選 |
| FINAL_TOP_K | 5 | RRF 後送入 LLM 的 chunk 數 |
| RRF_K | 60 | RRF 公式平滑常數 |
| EMBED_MODEL | nomic-embed-text | 向量模型 |
| GEN_MODEL | gemma4:e4b | 子查詢生成 + 回答生成 |
