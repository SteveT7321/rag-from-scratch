# 05 — Agentic RAG

## 狀態：待實作

## 核心問題

Naive RAG 是死板的管線：**無論問什麼，都固定取回 Top-4，然後丟給 LLM**。

問題：
- 問簡單問題（「BRZ 有幾個座位？」）→ 取 4 個 chunk 太浪費
- 問複雜問題（「比較 MT 和 AT 版本的差異，哪個更適合通勤？」）→ 4 個 chunk 不夠，需要多輪取回
- 取回的 chunk 完全不相關 → 照樣喂給 LLM，導致幻覺

## 解法：讓 LLM 自己決定

Agentic RAG 的核心思想：**LLM 不只是生成答案，也要決定如何取回**。

### 架構：ReAct（Reason + Act）

```
使用者問題
    │
    ▼
[LLM：思考] → 我需要什麼資訊？
    │
    ▼
[LLM：行動] → 呼叫 search("EyeSight 功能")
    │
    ▼
[取回結果]
    │
    ▼
[LLM：觀察] → 這個夠嗎？還需要什麼？
    │
    ├─ 夠了 → 生成最終答案
    │
    └─ 不夠 → 再次行動：search("預碰撞煞車")
                    │
                    ▼
                [繼續循環，最多 N 輪]
```

### 工具（Tools）設計

給 LLM 三個工具：

```python
tools = {
    "search": "在 BRZ 車型冊裡搜尋相關資訊，輸入：查詢字串",
    "lookup": "取得特定頁碼的完整內容，輸入：頁碼",
    "finish": "回答已足夠，輸出最終答案，輸入：答案文字",
}
```

### LLM 輸出格式（Prompt Engineering）

```
你是一個 RAG 代理人。你有以下工具可以使用：
- search(query): 搜尋車型冊
- lookup(page): 取得某頁完整內容
- finish(answer): 輸出最終答案

請用以下格式思考和行動：

思考：我需要了解...
行動：search("EyeSight 自動煞車")
觀察：[搜尋結果]
思考：我還需要...
行動：search("碰撞防護")
觀察：[搜尋結果]
思考：資訊已足夠。
行動：finish("EyeSight 系統包含...")
```

## 升級版：Self-RAG / CRAG

### Self-RAG（自我反思）

在取回之後，讓 LLM 自評：

```
取回的 chunks
    │
    ▼
[LLM 評分] 這些 chunks 和問題相關嗎？（分數 0-1）
    │
    ├─ 高相關 → 直接用
    ├─ 部分相關 → 過濾掉不相關的，用剩下的
    └─ 完全不相關 → 放棄取回結果，改用 LLM 自身知識回答（並標記）
```

### CRAG（Corrective RAG）

更進一步：若取回結果品質差，自動**重寫查詢**再試一次。

```
取回品質低
    │
    ▼
Query Rewriting（換個角度問）
    │
    ▼
再次取回
```

## 實作計畫

### 核心迴圈

```python
def agentic_rag(question: str, max_steps: int = 5) -> str:
    history = []
    
    for step in range(max_steps):
        # 讓 LLM 決定下一步行動
        action = llm_decide(question, history)
        
        if action["tool"] == "search":
            result = retrieve(action["query"], chunks, embeddings)
            history.append(("search", action["query"], result))
            
        elif action["tool"] == "lookup":
            result = get_page(action["page"])
            history.append(("lookup", action["page"], result))
            
        elif action["tool"] == "finish":
            return action["answer"]
    
    # 超過步數限制，強制回答
    return force_answer(question, history)
```

### 檔案結構

```
05_agentic_rag/
├── plan.md
├── requirements.txt
├── agent.py          # ReAct 主迴圈
├── tools.py          # search / lookup / finish 工具
├── prompts.py        # Prompt 模板
├── self_rag.py       # Self-RAG 自評估版本
└── agentic_rag.py    # CLI 整合
```

## 預期效果

| 問題類型 | Naive RAG | Agentic RAG |
|----------|-----------|-------------|
| 簡單問題 | 取 4 chunk（浪費） | 取 1 chunk 就夠，快速回答 |
| 複雜多步問題 | 一次取回不夠 | 多輪取回，逐步建立答案 |
| 取回失敗 | 幻覺 | 標記「找不到」或重試 |
| 需要比較兩個選項 | 可能只取到其中一個 | 分別 search 兩個選項 |

## 挑戰

- **Prompt 設計**：讓 gemma4 嚴格遵守輸出格式不容易
- **迴圈控制**：防止 LLM 陷入無限搜尋
- **延遲增加**：每步都需要 LLM 呼叫，比 Naive RAG 慢 3-5 倍

## 參考

- ReAct 論文：ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022)
- Self-RAG 論文：Self-RAG: Learning to Retrieve, Generate, and Critique (Asai et al., 2023)
- CRAG 論文：Corrective Retrieval Augmented Generation (Yan et al., 2024)
