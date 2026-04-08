"""
純手刻 RAG 系統 - Subaru BRZ 台灣車型冊
技術棧: PyMuPDF + numpy + Ollama (nomic-embed-text + gemma4)

用法:
  python rag.py ingest brz_tw.pdf       # 建立索引
  python rag.py chat                     # 互動問答
  python rag.py chat --query "問題"      # 單次查詢
"""

import argparse
import io
import json
import pickle
import re
import sys

# Windows 終端 UTF-8 輸出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests

# ─── 設定 ───────────────────────────────────────────────────────────────────

OLLAMA_BASE = "http://localhost:11434"
EMBED_MODEL  = "nomic-embed-text"
GEN_MODEL    = "gemma4:e4b"
INDEX_DIR    = Path(__file__).parent / "index"
CHUNKS_FILE  = INDEX_DIR / "chunks.pkl"
EMBED_FILE   = INDEX_DIR / "embeddings.npy"

CHUNK_SIZE   = 500   # 最大字元數
CHUNK_OVERLAP = 50   # 前後重疊字元數
TOP_K        = 4     # 取回最相似 chunks 數量

PROMPT_TEMPLATE = """\
你是一位熟悉 Subaru BRZ 的專家。請根據以下資料回答問題。
若資料中找不到答案，請直接說「資料中沒有相關資訊」，不要猜測。

[參考資料]
{context}

[問題]
{question}

[回答]"""

# ─── PDF 解析 ────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    """
    回傳 list of dict: {"page": int, "text": str}
    """
    import fitz  # pymupdf

    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        text = text.strip()
        if text:
            pages.append({"page": page_num + 1, "text": text})
    doc.close()
    return pages


# ─── Chunking ────────────────────────────────────────────────────────────────

def split_text(text: str, max_chars: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    先依段落切，超過 max_chars 的段落再細切，保留 overlap。
    """
    # 清理多餘空白
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 先按段落切
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    for para in paragraphs:
        if len(para) <= max_chars:
            chunks.append(para)
        else:
            # 細切超長段落
            start = 0
            while start < len(para):
                end = start + max_chars
                chunk = para[start:end]
                chunks.append(chunk)
                start = end - overlap

    return [c for c in chunks if len(c.strip()) > 10]


def build_chunks(pages: List[Dict]) -> List[Dict]:
    """
    回傳 list of dict: {"page": int, "text": str, "chunk_id": int}
    """
    all_chunks = []
    chunk_id = 0
    for page_info in pages:
        texts = split_text(page_info["text"])
        for t in texts:
            all_chunks.append({
                "chunk_id": chunk_id,
                "page": page_info["page"],
                "text": t,
            })
            chunk_id += 1
    return all_chunks


# ─── Embedding ───────────────────────────────────────────────────────────────

def embed_text(text: str) -> List[float]:
    """呼叫 Ollama /api/embed，回傳單一向量。"""
    resp = requests.post(
        f"{OLLAMA_BASE}/api/embed",
        json={"model": EMBED_MODEL, "input": text},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    # Ollama 回傳格式: {"embeddings": [[...]]}
    return data["embeddings"][0]


def embed_batch(chunks: List[Dict]) -> np.ndarray:
    """
    批次 embed 所有 chunks，顯示進度。
    回傳 shape (N, dim) 的 numpy 陣列。
    """
    n = len(chunks)
    vectors = []
    for i, chunk in enumerate(chunks):
        # 手刻進度條
        pct = (i + 1) / n
        bar_len = 30
        filled = int(bar_len * pct)
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"\r  [{bar}] {i+1}/{n}", end="", flush=True)

        vec = embed_text(chunk["text"])
        vectors.append(vec)

    print()  # 換行
    return np.array(vectors, dtype=np.float32)


# ─── 向量存儲 ─────────────────────────────────────────────────────────────────

def save_index(chunks: List[Dict], embeddings: np.ndarray) -> None:
    INDEX_DIR.mkdir(exist_ok=True)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)
    np.save(str(EMBED_FILE), embeddings)


def load_index() -> Tuple[List[Dict], np.ndarray]:
    if not CHUNKS_FILE.exists() or not EMBED_FILE.exists():
        print("錯誤：找不到索引。請先執行: python rag.py ingest <pdf_path>")
        sys.exit(1)
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    embeddings = np.load(str(EMBED_FILE))
    return chunks, embeddings


# ─── 相似度搜尋 ───────────────────────────────────────────────────────────────

def cosine_similarity(query_vec: np.ndarray, embed_matrix: np.ndarray) -> np.ndarray:
    """回傳 query 與每個 row 的餘弦相似度。"""
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    norms = np.linalg.norm(embed_matrix, axis=1, keepdims=True) + 1e-10
    normed = embed_matrix / norms
    return normed @ q


def retrieve(query: str, chunks: List[Dict], embeddings: np.ndarray, k: int = TOP_K) -> List[Dict]:
    query_vec = np.array(embed_text(query), dtype=np.float32)
    scores = cosine_similarity(query_vec, embeddings)
    top_indices = np.argsort(scores)[::-1][:k]
    results = []
    for idx in top_indices:
        chunk = chunks[idx].copy()
        chunk["score"] = float(scores[idx])
        results.append(chunk)
    return results


# ─── 生成 ─────────────────────────────────────────────────────────────────────

def generate(query: str, context_chunks: List[Dict]) -> None:
    """呼叫 Ollama 生成回答，串流輸出到終端。"""
    context_parts = []
    for i, c in enumerate(context_chunks, 1):
        context_parts.append(f"[第{c['page']}頁]\n{c['text']}")
    context = "\n\n".join(context_parts)

    prompt = PROMPT_TEMPLATE.format(context=context, question=query)

    resp = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json={"model": GEN_MODEL, "prompt": prompt, "stream": True},
        stream=True,
        timeout=120,
    )
    resp.raise_for_status()

    for line in resp.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        token = data.get("response", "")
        print(token, end="", flush=True)
        if data.get("done"):
            break
    print()  # 換行


# ─── CLI 命令 ─────────────────────────────────────────────────────────────────

def cmd_ingest(args: argparse.Namespace) -> None:
    pdf_path = args.pdf
    if not Path(pdf_path).exists():
        print(f"錯誤：找不到檔案 {pdf_path}")
        sys.exit(1)

    print(f"[1/4] 解析 PDF: {pdf_path}")
    pages = extract_text_from_pdf(pdf_path)
    print(f"      共 {len(pages)} 頁有文字")

    print("[2/4] 切割 chunks")
    chunks = build_chunks(pages)
    print(f"      共 {len(chunks)} 個 chunks")

    print(f"[3/4] 生成 embeddings（模型: {EMBED_MODEL}）")
    embeddings = embed_batch(chunks)
    print(f"      向量維度: {embeddings.shape}")

    print(f"[4/4] 儲存索引至 {INDEX_DIR}/")
    save_index(chunks, embeddings)

    print("\n完成！可以開始問答了：")
    print("  python rag.py chat")
    print('  python rag.py chat --query "BRZ 馬力是多少？"')


def cmd_chat(args: argparse.Namespace) -> None:
    print("載入索引...", end=" ", flush=True)
    chunks, embeddings = load_index()
    print(f"OK（{len(chunks)} chunks）")

    if args.query:
        # 單次查詢模式
        _answer(args.query, chunks, embeddings)
    else:
        # 互動模式
        print(f'\n互動問答模式（輸入 "exit" 離開）')
        print("─" * 50)
        while True:
            try:
                query = input("\n你：").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再見！")
                break
            if not query:
                continue
            if query.lower() in ("exit", "quit", "bye", "掰掰"):
                print("再見！")
                break
            _answer(query, chunks, embeddings)


def _answer(query: str, chunks: List[Dict], embeddings: np.ndarray) -> None:
    print(f"\n搜尋相關資料...", end=" ", flush=True)
    results = retrieve(query, chunks, embeddings)
    print(f"找到 {len(results)} 個相關片段")

    # 顯示來源頁碼
    pages_cited = sorted(set(r["page"] for r in results))
    print(f"參考頁碼：第 {', '.join(str(p) for p in pages_cited)} 頁")
    print(f"\nBRZ 專家：", end="", flush=True)

    generate(query, results)
    print("─" * 50)


# ─── 主程式 ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="純手刻 RAG - Subaru BRZ 台灣車型冊問答系統"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ingest 子命令
    p_ingest = subparsers.add_parser("ingest", help="解析 PDF 並建立向量索引")
    p_ingest.add_argument("pdf", help="PDF 檔案路徑")
    p_ingest.set_defaults(func=cmd_ingest)

    # chat 子命令
    p_chat = subparsers.add_parser("chat", help="問答模式")
    p_chat.add_argument("--query", "-q", help="直接提問（不加則進入互動模式）")
    p_chat.set_defaults(func=cmd_chat)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
