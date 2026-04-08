"""
02 — Advanced RAG
三種升級技術：HyDE、Query Rewriting、LLM Re-ranking
索引預設共用 01_naive_rag/index/，不需要重新 embed。

用法:
  python advanced_rag.py ingest ../brz_tw.pdf     # 若要自建索引
  python advanced_rag.py chat                      # 互動（預設用 01 的索引）
  python advanced_rag.py chat -q "問題" --hyde
  python advanced_rag.py chat -q "問題" --rewrite
  python advanced_rag.py chat -q "問題" --rerank
  python advanced_rag.py chat -q "問題" --all     # 三種全開
"""

import argparse
import io
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ─── 設定 ────────────────────────────────────────────────────────────────────

OLLAMA_BASE  = "http://localhost:11434"
EMBED_MODEL  = "nomic-embed-text"
GEN_MODEL    = "gemma4:e4b"
CHUNK_SIZE   = 500
CHUNK_OVERLAP = 50
TOP_K        = 4
RERANK_POOL  = 12   # Re-ranking 前先取幾個候選

# 預設共用 01 的索引
SHARED_INDEX = Path(__file__).resolve().parent.parent / "01_naive_rag" / "index"
LOCAL_INDEX  = Path(__file__).resolve().parent / "index"

PROMPT_TEMPLATE = """\
你是一位熟悉 Subaru BRZ 的專家。請根據以下資料回答問題。
若資料中找不到答案，請直接說「資料中沒有相關資訊」，不要猜測。

[參考資料]
{context}

[問題]
{question}

[回答]"""

# ─── Ollama 基礎呼叫 ──────────────────────────────────────────────────────────

def embed_text(text: str) -> List[float]:
    resp = requests.post(
        f"{OLLAMA_BASE}/api/embed",
        json={"model": EMBED_MODEL, "input": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def generate_text(prompt: str, stream: bool = False) -> str:
    resp = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json={"model": GEN_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def stream_generate(prompt: str) -> None:
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
        print(data.get("response", ""), end="", flush=True)
        if data.get("done"):
            break
    print()

# ─── PDF & Chunking（與 01 相同）─────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    import fitz
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        text = doc[i].get_text("text").strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages


def split_text(text: str) -> List[str]:
    text = re.sub(r"\n{3,}", "\n\n", text)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for para in paragraphs:
        if len(para) <= CHUNK_SIZE:
            chunks.append(para)
        else:
            start = 0
            while start < len(para):
                chunks.append(para[start:start + CHUNK_SIZE])
                start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if len(c.strip()) > 10]


def build_chunks(pages: List[Dict]) -> List[Dict]:
    all_chunks, chunk_id = [], 0
    for p in pages:
        for t in split_text(p["text"]):
            all_chunks.append({"chunk_id": chunk_id, "page": p["page"], "text": t})
            chunk_id += 1
    return all_chunks


def embed_batch(chunks: List[Dict]) -> np.ndarray:
    n = len(chunks)
    vectors = []
    for i, chunk in enumerate(chunks):
        bar = "#" * int(30 * (i + 1) / n) + "-" * (30 - int(30 * (i + 1) / n))
        print(f"\r  [{bar}] {i+1}/{n}", end="", flush=True)
        vectors.append(embed_text(chunk["text"]))
    print()
    return np.array(vectors, dtype=np.float32)

# ─── 索引 I/O ────────────────────────────────────────────────────────────────

def save_index(chunks, embeddings, index_dir: Path):
    index_dir.mkdir(exist_ok=True)
    with open(index_dir / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    np.save(str(index_dir / "embeddings.npy"), embeddings)


def load_index(index_dir: Path) -> Tuple[List[Dict], np.ndarray]:
    if not (index_dir / "chunks.pkl").exists():
        print(f"錯誤：找不到索引 {index_dir}")
        sys.exit(1)
    with open(index_dir / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    embeddings = np.load(str(index_dir / "embeddings.npy"))
    return chunks, embeddings

# ─── 基礎向量搜尋 ─────────────────────────────────────────────────────────────

def cosine_search(query_vec: np.ndarray, embeddings: np.ndarray, k: int) -> List[int]:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    scores = (embeddings / norms) @ q
    return list(np.argsort(scores)[::-1][:k])


def retrieve_base(query: str, chunks: List[Dict], embeddings: np.ndarray, k: int) -> List[Dict]:
    vec = np.array(embed_text(query), dtype=np.float32)
    idxs = cosine_search(vec, embeddings, k)
    return [dict(chunks[i], score=0.0) for i in idxs]

# ─── 技術一：HyDE ─────────────────────────────────────────────────────────────

def hyde(question: str) -> str:
    """
    生成假設性答案，用假設答案的 embedding 去搜尋，
    因為假設答案的措辭更接近文件本身。
    """
    prompt = (
        "請用繁體中文寫一個簡短的假設性回答（2-4句），"
        "假設你在回答以下問題，且答案可以在 Subaru BRZ 車型冊中找到。"
        "只寫回答內容，不要加說明或標題。\n\n"
        f"問題：{question}"
    )
    return generate_text(prompt)


def retrieve_hyde(question: str, chunks: List[Dict], embeddings: np.ndarray) -> List[Dict]:
    hypo = hyde(question)
    print(f"  [HyDE 假設答案] {hypo[:80]}...")
    vec = np.array(embed_text(hypo), dtype=np.float32)
    idxs = cosine_search(vec, embeddings, TOP_K)
    return [dict(chunks[i], score=0.0) for i in idxs]

# ─── 技術二：Query Rewriting ──────────────────────────────────────────────────

def rewrite_queries(question: str, n: int = 3) -> List[str]:
    """
    用 LLM 把一個問題改寫成 N 個不同角度的查詢，
    覆蓋更多可能的措辭和面向。
    """
    prompt = (
        f"請為以下問題生成 {n} 個不同角度的搜尋查詢，"
        "用來在 Subaru BRZ 車型冊裡搜尋。\n"
        "每行一個查詢，不要編號，不要其他說明。\n\n"
        f"原始問題：{question}"
    )
    text = generate_text(prompt)
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    queries = [question] + lines[:n]
    return list(dict.fromkeys(queries))  # 去重保序


def retrieve_rewrite(question: str, chunks: List[Dict], embeddings: np.ndarray) -> List[Dict]:
    queries = rewrite_queries(question, n=3)
    print(f"  [Rewrite 子查詢]")
    for q in queries:
        print(f"    • {q}")

    # 收集所有查詢的結果，去重
    seen, results = set(), []
    for q in queries:
        vec = np.array(embed_text(q), dtype=np.float32)
        idxs = cosine_search(vec, embeddings, TOP_K)
        for idx in idxs:
            if idx not in seen:
                seen.add(idx)
                results.append(chunks[idx])
    return results[:TOP_K * 2]  # 回傳更多候選給 re-ranking 或直接截取

# ─── 技術三：LLM Re-ranking ───────────────────────────────────────────────────

def rerank(question: str, candidates: List[Dict], top_k: int = TOP_K) -> List[Dict]:
    """
    讓 LLM 對每個候選 chunk 打分（0-10），
    比 cosine similarity 更精確，能理解語意關聯。
    """
    scored = []
    for chunk in candidates:
        prompt = (
            "請評估以下文字片段與問題的相關性，"
            "回傳一個 0 到 10 的整數分數，只回傳數字，不要其他說明。\n\n"
            f"問題：{question}\n"
            f"文字片段：{chunk['text'][:300]}\n\n"
            "相關性分數（0-10）："
        )
        try:
            resp = generate_text(prompt)
            score = float(re.search(r"\d+", resp).group())
            score = min(10.0, max(0.0, score))
        except Exception:
            score = 0.0
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]

# ─── 問答主流程 ───────────────────────────────────────────────────────────────

def answer(question: str, chunks: List[Dict], embeddings: np.ndarray,
           use_hyde: bool, use_rewrite: bool, use_rerank: bool) -> None:
    print(f"\n搜尋相關資料...", flush=True)

    if use_hyde:
        candidates = retrieve_hyde(question, chunks, embeddings)
    elif use_rewrite:
        candidates = retrieve_rewrite(question, chunks, embeddings)
    else:
        candidates = retrieve_base(question, chunks, embeddings, RERANK_POOL if use_rerank else TOP_K)

    if use_rerank:
        if not use_hyde and not use_rewrite:
            # 先取更多候選
            candidates = retrieve_base(question, chunks, embeddings, RERANK_POOL)
        print(f"  [Re-ranking {len(candidates)} 個候選...]")
        candidates = rerank(question, candidates, TOP_K)

    final = candidates[:TOP_K]
    pages = sorted(set(c["page"] for c in final))
    print(f"參考頁碼：第 {', '.join(str(p) for p in pages)} 頁")

    context = "\n\n".join(f"[第{c['page']}頁]\n{c['text']}" for c in final)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    print(f"\nBRZ 專家：", end="", flush=True)
    stream_generate(prompt)
    print("─" * 50)

# ─── CLI ─────────────────────────────────────────────────────────────────────

def cmd_ingest(args):
    pdf_path = args.pdf
    if not Path(pdf_path).exists():
        print(f"錯誤：找不到 {pdf_path}")
        sys.exit(1)
    print(f"[1/4] 解析 PDF: {pdf_path}")
    pages = extract_text_from_pdf(pdf_path)
    print(f"      共 {len(pages)} 頁")
    print("[2/4] 切割 chunks")
    ch = build_chunks(pages)
    print(f"      共 {len(ch)} 個 chunks")
    print(f"[3/4] 生成 embeddings")
    emb = embed_batch(ch)
    print(f"[4/4] 儲存索引")
    save_index(ch, emb, LOCAL_INDEX)
    print("完成！")


def cmd_chat(args):
    # 優先用本地索引，否則用 01 的共用索引
    index_dir = LOCAL_INDEX if LOCAL_INDEX.exists() else SHARED_INDEX
    if not index_dir.exists():
        print("找不到索引。請先執行: python advanced_rag.py ingest ../brz_tw.pdf")
        sys.exit(1)

    use_hyde    = args.hyde or args.all
    use_rewrite = args.rewrite or args.all
    use_rerank  = args.rerank or args.all

    print(f"載入索引 ({index_dir.name})...", end=" ")
    chunks, embeddings = load_index(index_dir)
    print(f"OK（{len(chunks)} chunks）")

    modes = []
    if use_hyde:    modes.append("HyDE")
    if use_rewrite: modes.append("Query Rewriting")
    if use_rerank:  modes.append("Re-ranking")
    if modes:
        print(f"啟用技術：{', '.join(modes)}")

    if args.query:
        answer(args.query, chunks, embeddings, use_hyde, use_rewrite, use_rerank)
    else:
        print(f'\n互動問答模式（"exit" 離開）\n' + "─" * 50)
        while True:
            try:
                q = input("\n你：").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再見！"); break
            if not q: continue
            if q.lower() in ("exit", "quit", "bye", "掰掰"):
                print("再見！"); break
            answer(q, chunks, embeddings, use_hyde, use_rewrite, use_rerank)


def main():
    parser = argparse.ArgumentParser(description="Advanced RAG — HyDE / Rewrite / Re-ranking")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("pdf")
    p_ingest.set_defaults(func=cmd_ingest)

    p_chat = sub.add_parser("chat")
    p_chat.add_argument("--query", "-q")
    p_chat.add_argument("--hyde",    action="store_true", help="使用 HyDE")
    p_chat.add_argument("--rewrite", action="store_true", help="使用 Query Rewriting")
    p_chat.add_argument("--rerank",  action="store_true", help="使用 LLM Re-ranking")
    p_chat.add_argument("--all",     action="store_true", help="三種技術全開")
    p_chat.set_defaults(func=cmd_chat)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
