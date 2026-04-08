# RAG 系統建置計畫

## Context

為 `rag/` 子目錄建立一個純手刻的 RAG（Retrieval-Augmented Generation）系統，
目標文件為 Subaru BRZ 台灣市場車型冊（繁體中文 PDF，11頁）。

技術棧：
- Python 3.8.8（無 LangChain/LlamaIndex 框架）
- Ollama 本地模型（已安裝 gemma4）
- CLI 介面（argparse）
- 向量存儲：numpy（純手刻，無 FAISS/ChromaDB）

---

## 目標檔案結構

```
rag/
├── rag.py           ← 主程式（唯一 Python 檔案）
├── requirements.txt
├── brz_tw.pdf       ← 已存在
└── index/           ← 執行 ingest 後自動產生
    ├── chunks.pkl   ← 文字 chunks
    └── embeddings.npy ← embedding 向量
```

---

## 系統架構（rag.py 內部模組）

```
ingest 子命令:
  PDF → 提取文字 → chunking → Ollama embedding → 儲存 index

chat 子命令:
  query → Ollama embedding → cosine similarity → top-k chunks → Ollama generate → 回答
```

---

## 實作細節

### 1. PDF 解析
- 使用 `pymupdf`（`import fitz`）逐頁提取文字
- 過濾空白頁、處理繁體中文排版

### 2. Chunking（純手刻）
- 按段落切分（`\n\n` 分隔）
- 超過 max_chars（預設 500）的段落再切，保留 overlap（50字）
- 每個 chunk 記錄來源頁碼

### 3. Embedding
- Ollama API：`POST http://localhost:11434/api/embed`
- 模型：`nomic-embed-text`（需 `ollama pull nomic-embed-text`）
- 批次處理，顯示進度條（手刻 tqdm-like）

### 4. 向量存儲
- embeddings → `numpy.ndarray`，shape = (N, dim)
- 存為 `index/embeddings.npy`（numpy save）
- chunks metadata → `index/chunks.pkl`（pickle）

### 5. 相似度搜尋
- 查詢向量 L2 正規化後，與所有向量做 dot product
- 回傳 top-k（預設 k=4）最相似 chunks

### 6. 生成
- Ollama API：`POST http://localhost:11434/api/generate`
- 模型：`gemma4`（已安裝）
- Prompt 模板（繁體中文）：
  ```
  你是一位熟悉 Subaru BRZ 的專家。請根據以下資料回答問題。
  若資料中找不到答案，請直接說不知道。

  [參考資料]
  {context}

  [問題]
  {question}
  ```
- stream=True，逐字輸出到終端

### 7. CLI
```bash
# 建立索引
python rag.py ingest brz_tw.pdf

# 互動問答
python rag.py chat

# 單次查詢
python rag.py chat --query "BRZ 的馬力是多少？"
```

---

## requirements.txt

```
pymupdf>=1.24.0
numpy>=1.24.0
```

（其他都用標準庫：`requests`, `pickle`, `argparse`, `pathlib`）

---

## 關鍵 API 細節

### Ollama Embed API
```
POST http://localhost:11434/api/embed
Body: {"model": "nomic-embed-text", "input": "text here"}
Response: {"embeddings": [[0.1, 0.2, ...]]}
```

### Ollama Generate API  
```
POST http://localhost:11434/api/generate
Body: {"model": "gemma4", "prompt": "...", "stream": true}
Response: 多行 JSON，每行有 {"response": "token", "done": false}
```

---

## 驗證方式

1. `pip install -r requirements.txt`
2. `ollama pull nomic-embed-text`
3. `python rag.py ingest brz_tw.pdf` → 應看到 index 目錄產生
4. `python rag.py chat --query "BRZ 有哪些顏色選擇？"` → 應回傳車型冊中的顏色資訊
5. `python rag.py chat --query "EyeSight 系統有什麼功能？"` → 應回傳安全系統說明
6. `python rag.py chat` → 互動模式，可持續問答

---

## 注意事項

- Ollama embed 端點：新版用 `/api/embed`（非 `/api/embeddings`），需確認版本
- Python 3.8 相容：避免 walrus operator、match 語法，型別標注用 `typing` 模組
- 繁體中文 chunking：以字元數而非 token 數計算 chunk 大小
