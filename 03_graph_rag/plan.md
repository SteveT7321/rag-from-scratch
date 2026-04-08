# 03 — Graph RAG

## 狀態：待實作

## 核心問題

向量搜尋只能找「語意相近的片段」，但無法回答需要**推理多個實體關係**的問題。

例如：
- 「BRZ 的引擎和底盤分別來自哪個合作廠商？」
- 「EyeSight 和 VDC 各在什麼情況下啟動？兩者有關聯嗎？」

這類問題需要連接多個節點才能回答，純向量搜尋做不到。

## 什麼是 Graph RAG

Microsoft 2024 年提出，核心思路：

1. 把文件裡的**實體（Entity）**和**關係（Relation）**抽取出來，建成知識圖譜
2. 查詢時，先找到相關實體，再沿圖的邊走到相關節點，取回上下文
3. 可以做**多跳推理（Multi-hop Reasoning）**

```
文件
  │
  ▼
[LLM 抽取] → 實體：BRZ、EyeSight、Subaru、Boxer 引擎...
            → 關係：BRZ --搭載--> EyeSight
                    BRZ --使用--> Boxer 引擎
                    EyeSight --功能包括--> 自動煞車
  │
  ▼
建立圖（Nodes + Edges）
  │
  ▼
查詢："EyeSight 能防止哪些事故？"
  │
  ▼
找到 EyeSight 節點 → 沿邊找相關節點 → 取回片段
  │
  ▼
LLM 生成回答
```

## 兩種 Graph RAG 模式

### Local Search（局部搜尋）
從查詢相關的實體出發，沿鄰近邊展開，適合具體問題。

### Global Search（全域搜尋）
把圖的所有社群（Community）做摘要，適合「這份文件在講什麼」的宏觀問題。

本專案先實作 **Local Search**。

## 實作計畫

### 階段一：Entity & Relation 抽取

用 LLM（gemma4）從每個 chunk 抽取三元組：

```
輸入：chunk 文字
輸出：[(主體, 關係, 客體), ...]

範例：
  ("BRZ", "搭載安全系統", "EyeSight")
  ("EyeSight", "功能包括", "預碰撞煞車")
  ("BRZ 2.4 MT", "使用變速箱", "6速手排")
```

### 階段二：建立圖結構

```python
# 用 Python 標準 dict 純手刻（不依賴 networkx）
graph = {
    "nodes": {
        "BRZ": {"type": "車款", "mentions": 15},
        "EyeSight": {"type": "安全系統", "mentions": 8},
        ...
    },
    "edges": [
        {"from": "BRZ", "rel": "搭載", "to": "EyeSight", "source_chunk": 5},
        ...
    ]
}
```

存為 `index/graph.json`

### 階段三：圖搜尋

```
查詢問題
    │
    ▼
找出問題提到的實體（字串比對 + embedding）
    │
    ▼
BFS/DFS 展開 1-2 跳的鄰居節點
    │
    ▼
收集這些節點的 source_chunk，組合成 context
    │
    ▼
LLM 生成回答
```

### 檔案結構

```
03_graph_rag/
├── plan.md
├── requirements.txt
├── extract.py       # LLM 抽取三元組
├── build_graph.py   # 建立圖結構
├── search.py        # 圖搜尋邏輯
├── graph_rag.py     # CLI 整合
└── index/
    ├── chunks.pkl
    ├── embeddings.npy
    └── graph.json   ← 新增
```

### 建議套件

```
pymupdf==1.23.8
numpy
requests
```

（全部用標準庫 + numpy，不裝 networkx）

## 挑戰與注意事項

- **抽取品質依賴 LLM**：gemma4 中文抽取能力需實測
- **圖會有雜訊**：LLM 有時抽出不存在的關係，需過濾低頻邊
- **BRZ 車型冊圖譜規模小**：預估 30-50 個實體，適合驗證概念

## 參考

- Microsoft GraphRAG 論文：From Local to Global (Edge et al., 2024)
- Microsoft GraphRAG GitHub：microsoft/graphrag
