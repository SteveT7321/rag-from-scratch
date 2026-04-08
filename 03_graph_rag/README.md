# 03 — Graph RAG

從文件中抽取實體與關係三元組，建成知識圖譜。
查詢時先向量搜尋找種子節點，再 BFS 展開圖，取回更廣的相關 chunks。
適合處理需要多實體關聯推理的問題。

---

## 原理

### 流程

```
ingest 階段：
PDF → 文字切塊 → LLM 抽取三元組 → 建知識圖譜 graph.json
                 ↓
             embed → embeddings.npy

查詢階段：
問題 → embed → 向量搜尋 Top-4（種子 chunks）
                 ↓
    找出種子 chunks 裡的所有實體
                 ↓
    BFS 展開 1 跳 → 取得相鄰實體
                 ↓
    收集所有相關實體對應的 chunks → 重新用向量分數排序
                 ↓
    Top-8 chunks → LLM 生成回答
```

### 三元組格式

```
主體 | 關係 | 客體
```

Prompt 要求 LLM 只抽取文字中明確存在的事實，例如：
```
SUBARU BRZ | 具備 | 更強大的馬力
EyeSight   | 配備 | 立體攝影機
LED 頭燈   | 提供 | 日間或夜晚的明亮視野
```

### 為何優於純向量搜尋

向量搜尋：找到「EyeSight」相關的直接段落
Graph RAG：從「EyeSight」出發，BFS 展開找到「安全特色」「全方位安全性」等相鄰節點，
           再取回這些節點對應的 chunks，召回更多相關資訊

---

## 使用方式

```bash
# 1. 建立索引（含知識圖譜，約需 5-10 分鐘）
python graph_rag.py ingest ../brz_tw.pdf

# 2. 查詢
python graph_rag.py chat -q "EyeSight 有哪些安全功能？"

# 3. 查看知識圖譜概覽
python graph_rag.py show-graph

# 4. 互動模式
python graph_rag.py chat
```

---

## 實測結果

### 知識圖譜統計（brz_tw.pdf，25 chunks）

| 項目 | 數值 |
|------|------|
| 節點數（實體） | 360 |
| 邊數（關係） | 257 |
| ingest 時間 | 約 8 分鐘 |

### 高頻實體 Top-10

| 實體 | 出現次數 | 關聯 chunks |
|------|---------|------------|
| 車輛 | 14 | 5 個 |
| 引擎 | 13 | 3 個 |
| SUBARU BRZ | 11 | 4 個 |
| EyeSight | 7 | 2 個 |
| BRZ 2.4 AT | 7 | 2 個 |
| BRZ 2.4 MT | 7 | 2 個 |

### 關係樣本

```
SUBARU BRZ --[具備]--> 更強大的馬力
SUBARU BRZ --[具備]--> 卓越的操控性
LED 頭燈   --[提供]--> 日間或夜晚的明亮視野
雙出式排氣管 --[對應]--> 強大的引擎輸出
EyeSight   --[如同]--> 全時監控前方道路狀況的第二雙眼
```

---

### 查詢測試

**「EyeSight 有哪些安全功能？」**

向量搜尋初始 chunks：[20, 16, 0, 15]
BFS 展開新增實體：安全特色、全方位安全性、Apple CarPlay & Android Auto…
參考頁碼：第 2, 5, 7, 10 頁（共 8 個片段）

✅ 成功回答：
> EyeSight 配備兩顆立體攝影機，能捕捉 3D 彩色影像，辨識車輛、摩托車、自行車、行人等潛在危險，並主動警示駕駛。

> Naive RAG 對此問題完全失敗（EyeSight 資訊分散在多個 chunk），Graph RAG 透過圖遍歷成功聚合。

---

**「BRZ 的引擎馬力是多少？」**

向量搜尋初始 chunks：[0, 15, 17, 1]
BFS 展開新增實體：EyeSight 功能、2.4 升水平對臥自然進氣引擎…
參考頁碼：第 2, 3, 4, 6, 7, 10 頁（共 8 個片段）

⚠️ 部分正確：回傳了多個數值（含規格表中的非馬力數字），答案有雜訊。
BFS 展開了太多不相關的 chunk，LLM 在龐雜的上下文中提取數值出錯。

---

**「BRZ 有哪些駕駛模式？」**

向量搜尋命中外觀/座艙相關 chunks，BFS 未能擴展到含駕駛模式的頁面。
❌ 回答：資料中沒有相關資訊

（圖譜中有 TRACK 賽道模式、SPORT 運動模式節點，但種子 chunks 不夠近，BFS 1 跳未能到達）

---

## 效果分析

### 優點
- **多跳資訊聚合**：透過圖遍歷，可以找到「EyeSight → 安全特色 → 相關 chunks」等間接連結
- **關係型問題**：「A 和 B 有什麼關聯？」類型的問題特別適合
- **知識圖譜可視化**：`show-graph` 可直接查看抽取的實體與關係網絡

### 缺點
- **BFS 展開過多雜訊**：1 跳展開可能引入不相關 chunk，干擾 LLM 的回答
- **三元組抽取品質依賴 LLM**：gemma4 偶爾產生格式錯誤或語意不精確的三元組
- **ingest 時間長**：每個 chunk 都需要 LLM 呼叫抽取三元組，25 chunks 約 8 分鐘
- **圖稀疏問題**：BRZ 車型冊文字層稀疏，三元組數量有限（257 條邊 / 360 個節點）

---

## 關鍵設定

| 參數 | 值 | 說明 |
|------|----|------|
| TOP_K_VEC | 4 | 向量搜尋初始 chunk 數 |
| HOP_LIMIT | 1 | BFS 展開跳數 |
| CHUNK_SIZE | 500 | 每個 chunk 最大字元數 |
| GEN_MODEL | gemma4:e4b | 三元組抽取 + 回答生成 |

---

## 索引結構

```
index/
├── chunks.pkl       # 25 個文字 chunks
├── embeddings.npy   # shape (25, 768)
└── graph.json       # 360 節點，257 條邊
```

graph.json 結構：
```json
{
  "nodes": { "實體名稱": {"count": 11, "chunk_ids": [0, 3, 7]} },
  "edges": [ {"from": "SUBARU BRZ", "rel": "具備", "to": "更強大的馬力", "chunk_id": 0} ]
}
```
