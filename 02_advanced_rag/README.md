# 02 — Advanced RAG

在 Naive RAG 基礎上加入三種升級技術：HyDE、Query Rewriting、LLM Re-ranking。
每種技術獨立可用，也可以組合使用（`--all`）。
索引共用 `01_naive_rag/index/`，不需要重新 embed。

---

## 三種技術原理

### 1. HyDE（Hypothetical Document Embedding）

**問題**：使用者提問的措辭與文件的表達方式不同，向量距離偏大導致搜尋失準。

**解法**：先讓 LLM 生成一個「假設性回答」，用假設答案的向量去搜尋，而非直接用問題向量。

```
原始問題 → LLM 生成假設答案（2-4句）
                ↓
         embed(假設答案) → 向量搜尋
```

假設答案的措辭（「搭載 2.4 升水平對置引擎，最大馬力 234ps」）更接近文件本身的語言，
因此向量距離更小，命中率更高。

---

### 2. Query Rewriting

**問題**：單一查詢只代表一種角度，可能錯過文件中用不同措辭表達的資訊。

**解法**：讓 LLM 將原始問題改寫成 N 個不同角度的子查詢，分別搜尋後合併去重。

```
原始問題 → LLM 生成 3 個子查詢
                ↓
         每個子查詢各自向量搜尋 → 結果合併去重 → Top-K
```

例：「BRZ 的引擎馬力？」→「2.4L 引擎馬力」、「BRZ 引擎輸出功率」、「BRZ 規格 馬力數值」

---

### 3. LLM Re-ranking

**問題**：餘弦相似度只衡量語意距離，無法精確判斷哪個 chunk 真的能回答問題。

**解法**：向量搜尋先取大量候選（12個），再讓 LLM 對每個 chunk 打出相關性分數（0-10），
依分數重新排序後取 Top-K 送入生成。

```
query → cosine 搜尋 Top-12 候選
              ↓
   LLM 逐一打分（0-10）→ 依分數重排 → Top-4
```

---

## 環境設置

```bash
# 索引共用 01，不需要另外建立
# 若 01 的索引已存在，直接執行：

cd 02_advanced_rag
python advanced_rag.py chat
```

---

## 使用方式

```bash
# 基礎模式（無升級技術，行為與 01 相同）
python advanced_rag.py chat -q "問題"

# HyDE 模式
python advanced_rag.py chat -q "問題" --hyde

# Query Rewriting 模式
python advanced_rag.py chat -q "問題" --rewrite

# LLM Re-ranking 模式
python advanced_rag.py chat -q "問題" --rerank

# 三種全開
python advanced_rag.py chat -q "問題" --all

# 互動模式（可搭配任意 flag）
python advanced_rag.py chat --hyde
```

---

## 實測結果

索引：25 chunks（共用 01_naive_rag/index/）

### 查詢：「BRZ 的引擎馬力是多少？」

| 模式 | 參考頁碼 | 回答 | 結果 |
|------|---------|------|------|
| Baseline | 2, 3, 10 | 234 ps / 7,000 rpm（含額外混淆數值） | ⚠️ 正確但有雜訊 |
| HyDE | 2, 4, 6, 10 | **172 kW (234 ps) / 7,000 rpm** | ✅ 乾淨準確 |
| Query Rewriting | 2, 3, 10 | **234 ps（7,000 rpm）** | ✅ 簡潔正確 |
| Re-ranking | 2, 6, 10 | **234 ps（7,000 rpm）** | ✅ 正確，頁碼更精準 |

HyDE 生成的假設答案：「本車搭載 2.0 升 Boxer 自然進氣引擎，最大馬力 211 匹…」
（數值不準確，但措辭命中了文件中的規格段落）

Query Rewriting 生成的子查詢：
- BRZ 的引擎馬力是多少？（原始）
- 2.4L 引擎馬力
- BRZ 引擎輸出功率
- BRZ 規格 馬力數值

### 查詢：「BRZ 搭載什麼引擎？」（HyDE）

HyDE 假設答案：「搭載可靠且高效的 2.0 升水平對置引擎…」（排氣量不對）
→ 但仍命中正確頁（2, 4, 6）
→ 最終回答：✅ **BRZ 搭載 2.4 升水平對臥自然進氣引擎**

### 查詢：「EyeSight 安全系統有哪些功能？」（Query Rewriting）

子查詢：EyeSight 系統功能列表、BRZ 安全功能介紹、BRZ 駕駛輔助系統詳解
→ 結果：❌ **資料中沒有相關資訊**
→ 原因：EyeSight 功能說明在 PDF 圖片層，文字層不含這些資訊

### 查詢：「BRZ 的車體尺寸規格？」（--all 三技術全開）

→ 結果：❌ **資料中沒有相關資訊**
→ 原因：尺寸規格在 PDF 圖表中，超出文字層範圍

---

## 效果分析

### HyDE
- **優點**：假設答案措辭接近文件，即使假設答案本身數值不精確，仍能命中正確 chunk
- **缺點**：每次查詢多一次 LLM 呼叫，延遲增加；偶爾假設答案方向偏差太大反而干擾搜尋

### Query Rewriting
- **優點**：覆蓋多個同義表達，對措辭差異的容忍度高；子查詢品質好時效果明顯
- **缺點**：多個子查詢各自 embed，API 呼叫次數 ×N；子查詢相似時收益有限

### LLM Re-ranking
- **優點**：理論上能理解語意關聯，超越餘弦相似度的表面匹配
- **缺點**：gemma4 作為 0-10 分的評分器穩定性有限；每個候選都需要一次 LLM 呼叫（12 次），延遲最高
- **生產建議**：換用 Cross-Encoder 模型（如 `BAAI/bge-reranker-v2-m3`）效果更好

---

## 共同限制

三種技術都無法解決的根本問題：**PDF 文字層缺失**。
車型冊中的顏色選項、EyeSight 功能、車體尺寸、配備表等資訊以圖片排版，
文字層不含這些內容，任何文字型 RAG 均無法命中。

→ 解法見 [04 Multimodal RAG](../04_multimodal_rag/)（Vision LLM 解讀圖表）

---

## 關鍵設定

| 參數 | 值 | 說明 |
|------|----|------|
| TOP_K | 4 | 最終送入 LLM 的 chunk 數 |
| RERANK_POOL | 12 | Re-ranking 前的候選池大小 |
| Query Rewriting n | 3 | 生成子查詢數量 |
| EMBED_MODEL | nomic-embed-text | 共用 01 的向量模型 |
| GEN_MODEL | gemma4:e4b | 生成 + 打分共用同一模型 |
