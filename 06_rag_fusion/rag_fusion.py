"""
06 — RAG Fusion + Reciprocal Rank Fusion
同一個問題從多角度生成子查詢，分別搜尋，用 RRF 融合排名。
比單一查詢召回率更高，對開放性問題效果顯著。

RRF 公式：score(d) = Σ 1/(k + rank_i(d))  (k=60)

用法:
  python rag_fusion.py ingest ../brz_tw.pdf         # 建立索引
  python rag_fusion.py chat                          # 互動模式
  python rag_fusion.py chat -q "問題"
  python rag_fusion.py chat -q "問題" -n 5          # 生成 5 個子查詢
  python rag_fusion.py chat -q "問題" --show-queries # 顯示子查詢
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

OLLAMA_BASE   = "http://localhost:11434"
EMBED_MODEL   = "nomic-embed-text"
GEN_MODEL     = "gemma4:e4b"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
DEFAULT_N     = 4   # 預設子查詢數量
TOP_K_PER_Q   = 6   # 每個子查詢取幾個候選
FINAL_TOP_K   = 5   # RRF 後取幾個送給 LLM
RRF_K         = 60  # RRF 公式中的 k 常數

SHARED_INDEX = Path(__file__).resolve().parent.parent / "01_naive_rag" / "index"
LOCAL_INDEX  = Path(__file__).resolve().parent / "index"

QUERY_GEN_PROMPT = """\
針對以下問題，生成 {n} 個不同角度的搜尋查詢，用來在 Subaru BRZ 台灣車型冊中搜尋。
要求：
- 每個查詢覆蓋不同面向（性能、外觀、安全、配備、規格等）
- 使用不同的措辭和關鍵字
- 每行一個查詢，不要編號，不要其他說明

原始問題：{question}

子查詢："""

ANSWER_PROMPT = """\
你是一位熟悉 Subaru BRZ 的專家。請根據以下資料回答問題。
若資料中找不到答案，請直接說「資料中沒有相關資訊」，不要猜測。

[參考資料]
{context}

[問題]
{question}

[回答]"""

# ─── Ollama 基礎呼叫 ──────────────────────────────────────────────────────────

def embed_text(text: str) -> List[float]:
    resp = requests.post(f"{OLLAMA_BASE}/api/embed",
        json={"model": EMBED_MODEL, "input": text}, timeout=60)
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def generate_text(prompt: str) -> str:
    resp = requests.post(f"{OLLAMA_BASE}/api/generate",
        json={"model": GEN_MODEL, "prompt": prompt, "stream": False}, timeout=120)
    resp.raise_for_status()
    return resp.json()["response"]


def stream_generate(prompt: str) -> None:
    resp = requests.post(f"{OLLAMA_BASE}/api/generate",
        json={"model": GEN_MODEL, "prompt": prompt, "stream": True},
        stream=True, timeout=120)
    resp.raise_for_status()
    for line in resp.iter_lines():
        if not line: continue
        data = json.loads(line)
        print(data.get("response", ""), end="", flush=True)
        if data.get("done"): break
    print()

# ─── PDF & Chunking ───────────────────────────────────────────────────────────

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

# ─── 索引 I/O ────────────────────────────────────────────────────────────────

def save_index(chunks, embeddings, index_dir: Path):
    index_dir.mkdir(exist_ok=True)
    with open(index_dir / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    np.save(str(index_dir / "embeddings.npy"), embeddings)


def load_index(index_dir: Path) -> Tuple[List[Dict], np.ndarray]:
    if not (index_dir / "chunks.pkl").exists():
        print(f"找不到索引 {index_dir}"); sys.exit(1)
    with open(index_dir / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    embeddings = np.load(str(index_dir / "embeddings.npy"))
    return chunks, embeddings

# ─── 子查詢生成 ───────────────────────────────────────────────────────────────

def generate_subqueries(question: str, n: int) -> List[str]:
    """用 LLM 生成 n 個不同角度的子查詢，加上原問題。"""
    prompt = QUERY_GEN_PROMPT.format(n=n, question=question)
    response = generate_text(prompt)
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    # 去掉編號前綴（如 "1. "、"- "）
    clean = []
    for line in lines:
        line = re.sub(r"^[\d\-\*\•\.]+\s*", "", line).strip()
        if line:
            clean.append(line)
    # 原始問題排第一，去重
    queries = [question] + clean[:n]
    seen, unique = set(), []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    return unique

# ─── 向量搜尋 ─────────────────────────────────────────────────────────────────

def vector_search(query: str, embeddings: np.ndarray, k: int) -> List[Tuple[int, float]]:
    """回傳 [(chunk_idx, score), ...]，依分數降序。"""
    vec = np.array(embed_text(query), dtype=np.float32)
    q = vec / (np.linalg.norm(vec) + 1e-10)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    scores = (embeddings / norms) @ q
    top_idxs = np.argsort(scores)[::-1][:k]
    return [(int(idx), float(scores[idx])) for idx in top_idxs]

# ─── Reciprocal Rank Fusion ───────────────────────────────────────────────────

def reciprocal_rank_fusion(
    result_lists: List[List[Tuple[int, float]]],
    k: int = RRF_K,
    final_k: int = FINAL_TOP_K
) -> List[Tuple[int, float]]:
    """
    輸入：多個 [(chunk_idx, score), ...] 列表
    輸出：RRF 融合後的 [(chunk_idx, rrf_score), ...]，依分數降序

    RRF 公式：RRF(d) = Σ_i  1 / (k + rank_i(d))
    """
    rrf_scores: Dict[int, float] = {}

    for ranked_list in result_lists:
        for rank, (chunk_idx, _) in enumerate(ranked_list, start=1):
            rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0.0) + 1.0 / (k + rank)

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:final_k]

# ─── 問答主流程 ───────────────────────────────────────────────────────────────

def answer(question: str, chunks: List[Dict], embeddings: np.ndarray,
           n_queries: int, show_queries: bool) -> None:

    # 1. 生成子查詢
    print(f"\n生成 {n_queries} 個子查詢...", end=" ", flush=True)
    queries = generate_subqueries(question, n_queries)
    print(f"OK（實際 {len(queries)} 個）")

    if show_queries:
        for i, q in enumerate(queries):
            label = "（原始）" if i == 0 else f"（子查詢 {i}）"
            print(f"  {label} {q}")

    # 2. 每個子查詢分別搜尋
    print(f"分別搜尋並融合...", end=" ", flush=True)
    all_result_lists = []
    for q in queries:
        results = vector_search(q, embeddings, TOP_K_PER_Q)
        all_result_lists.append(results)

    # 3. RRF 融合
    fused = reciprocal_rank_fusion(all_result_lists, k=RRF_K, final_k=FINAL_TOP_K)
    print(f"OK")

    # 4. 取出 chunks
    final_chunks = [chunks[idx] for idx, _ in fused]
    rrf_scores_map = {idx: score for idx, score in fused}

    pages = sorted(set(c["page"] for c in final_chunks))
    print(f"RRF Top-{len(final_chunks)} chunks：第 {', '.join(str(p) for p in pages)} 頁")

    if show_queries:
        print("  RRF 分數：" + "  ".join(
            f"chunk{idx}={score:.4f}" for idx, score in fused
        ))

    # 5. 生成回答
    context = "\n\n".join(
        f"[第{c['page']}頁（RRF分數：{rrf_scores_map[c['chunk_id']]:.4f}）]\n{c['text']}"
        for c in final_chunks
    )
    prompt = ANSWER_PROMPT.format(context=context, question=question)

    print(f"\nBRZ 專家：", end="", flush=True)
    stream_generate(prompt)
    print("─" * 50)

# ─── CLI ─────────────────────────────────────────────────────────────────────

def cmd_ingest(args):
    pdf_path = args.pdf
    if not Path(pdf_path).exists():
        print(f"錯誤：找不到 {pdf_path}"); sys.exit(1)
    print(f"[1/3] 解析 PDF")
    pages = extract_text_from_pdf(pdf_path)
    print(f"      共 {len(pages)} 頁")
    print("[2/3] 切割 chunks")
    ch = build_chunks(pages)
    print(f"      共 {len(ch)} 個 chunks")
    print("[3/3] 生成 embeddings")
    n = len(ch)
    vecs = []
    for i, c in enumerate(ch):
        bar = "#" * int(30*(i+1)/n) + "-" * (30 - int(30*(i+1)/n))
        print(f"\r  [{bar}] {i+1}/{n}", end="", flush=True)
        vecs.append(embed_text(c["text"]))
    print()
    emb = np.array(vecs, dtype=np.float32)
    save_index(ch, emb, LOCAL_INDEX)
    print("完成！")


def cmd_chat(args):
    index_dir = LOCAL_INDEX if LOCAL_INDEX.exists() else SHARED_INDEX
    if not index_dir.exists():
        print("找不到索引，請先執行: python rag_fusion.py ingest ../brz_tw.pdf")
        sys.exit(1)

    print(f"載入索引 ({index_dir.name})...", end=" ")
    chunks, embeddings = load_index(index_dir)
    print(f"OK（{len(chunks)} chunks）")
    print(f"模式：RAG Fusion（每次生成 {args.n_queries} 個子查詢，RRF k={RRF_K}）")

    if args.query:
        answer(args.query, chunks, embeddings, args.n_queries, args.show_queries)
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
            answer(q, chunks, embeddings, args.n_queries, args.show_queries)


def main():
    parser = argparse.ArgumentParser(description="RAG Fusion — 多查詢 + Reciprocal Rank Fusion")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("pdf")
    p_ingest.set_defaults(func=cmd_ingest)

    p_chat = sub.add_parser("chat")
    p_chat.add_argument("--query", "-q")
    p_chat.add_argument("--n-queries", "-n", type=int, default=DEFAULT_N,
                        help=f"生成幾個子查詢（預設 {DEFAULT_N}）")
    p_chat.add_argument("--show-queries", action="store_true",
                        help="顯示生成的子查詢和 RRF 分數")
    p_chat.set_defaults(func=cmd_chat)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
