"""
03 — Graph RAG
從文件抽取實體與關係，建成知識圖譜，用圖遍歷擴展取回範圍。
支援跨實體的多跳推理，比純向量搜尋更能處理關係型問題。

用法:
  python graph_rag.py ingest ../brz_tw.pdf   # 建立索引 + 圖譜
  python graph_rag.py chat                    # 互動問答
  python graph_rag.py chat -q "問題"
  python graph_rag.py show-graph              # 顯示知識圖譜概覽
"""

import argparse
import io
import json
import pickle
import re
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Set, Tuple

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
TOP_K_VEC    = 4    # 向量搜尋取幾個初始 chunk
HOP_LIMIT    = 1    # 圖遍歷幾跳
INDEX_DIR    = Path(__file__).parent / "index"

EXTRACT_PROMPT = """\
從以下關於 Subaru BRZ 的文字中，抽取明確的實體與關係三元組。
格式：主體|關係|客體
每行一個三元組，只包含文字中明確提到的事實，不要推測。
若沒有明確的關係，回傳空白。

文字：
{text}

三元組（每行一個）："""

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

# ─── 圖譜抽取 ─────────────────────────────────────────────────────────────────

def extract_triplets(text: str) -> List[Tuple[str, str, str]]:
    """從文字抽取 (主體, 關係, 客體) 三元組。"""
    prompt = EXTRACT_PROMPT.format(text=text[:600])
    try:
        response = generate_text(prompt)
    except Exception:
        return []

    triplets = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3 and all(parts):
            subj, rel, obj = parts
            # 過濾太長或明顯錯誤的
            if len(subj) < 50 and len(obj) < 50 and len(rel) < 30:
                triplets.append((subj, rel, obj))
    return triplets


def build_graph(chunks: List[Dict]) -> Dict:
    """
    回傳圖結構：
    {
      "nodes": { "實體名稱": {"count": int, "chunk_ids": [int, ...]} },
      "edges": [ {"from": str, "rel": str, "to": str, "chunk_id": int} ]
    }
    """
    graph = {"nodes": {}, "edges": []}
    n = len(chunks)

    for i, chunk in enumerate(chunks):
        bar = "#" * int(30 * (i + 1) / n) + "-" * (30 - int(30 * (i + 1) / n))
        print(f"\r  [{bar}] {i+1}/{n}", end="", flush=True)

        triplets = extract_triplets(chunk["text"])
        for subj, rel, obj in triplets:
            # 更新 nodes
            for entity in (subj, obj):
                if entity not in graph["nodes"]:
                    graph["nodes"][entity] = {"count": 0, "chunk_ids": []}
                graph["nodes"][entity]["count"] += 1
                if chunk["chunk_id"] not in graph["nodes"][entity]["chunk_ids"]:
                    graph["nodes"][entity]["chunk_ids"].append(chunk["chunk_id"])

            # 加入 edge
            graph["edges"].append({
                "from": subj, "rel": rel, "to": obj,
                "chunk_id": chunk["chunk_id"]
            })

    print()
    return graph

# ─── 索引 I/O ────────────────────────────────────────────────────────────────

def save_index(chunks, embeddings, graph):
    INDEX_DIR.mkdir(exist_ok=True)
    with open(INDEX_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    np.save(str(INDEX_DIR / "embeddings.npy"), embeddings)
    with open(INDEX_DIR / "graph.json", "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)


def load_index():
    for fname in ("chunks.pkl", "embeddings.npy", "graph.json"):
        if not (INDEX_DIR / fname).exists():
            print(f"錯誤：找不到 {INDEX_DIR / fname}，請先執行 ingest")
            sys.exit(1)
    with open(INDEX_DIR / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    embeddings = np.load(str(INDEX_DIR / "embeddings.npy"))
    with open(INDEX_DIR / "graph.json", "r", encoding="utf-8") as f:
        graph = json.load(f)
    return chunks, embeddings, graph

# ─── 圖搜尋 ───────────────────────────────────────────────────────────────────

def cosine_search(query_vec: np.ndarray, embeddings: np.ndarray, k: int) -> List[int]:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    scores = (embeddings / norms) @ q
    return list(np.argsort(scores)[::-1][:k])


def find_entities_in_chunks(chunk_ids: List[int], graph: Dict) -> Set[str]:
    """找出這些 chunks 裡提到的實體。"""
    entities = set()
    chunk_id_set = set(chunk_ids)
    for entity, info in graph["nodes"].items():
        if chunk_id_set & set(info["chunk_ids"]):
            entities.add(entity)
    return entities


def bfs_expand(seed_entities: Set[str], graph: Dict, hops: int) -> Set[str]:
    """從種子實體出發，BFS 展開 hops 跳，收集相鄰實體。"""
    visited = set(seed_entities)
    frontier = deque(seed_entities)

    for _ in range(hops):
        next_frontier = []
        while frontier:
            entity = frontier.popleft()
            for edge in graph["edges"]:
                neighbor = None
                if edge["from"] == entity:
                    neighbor = edge["to"]
                elif edge["to"] == entity:
                    neighbor = edge["from"]
                if neighbor and neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.append(neighbor)
        frontier = deque(next_frontier)

    return visited


def retrieve_graph(question: str, chunks: List[Dict],
                   embeddings: np.ndarray, graph: Dict) -> List[Dict]:
    """
    1. 向量搜尋取初始 chunks
    2. 找出這些 chunks 提到的實體
    3. BFS 展開圖，找到相鄰實體
    4. 收集相鄰實體對應的 chunks
    5. 合併所有 chunks
    """
    # Step 1: 向量搜尋
    vec = np.array(embed_text(question), dtype=np.float32)
    seed_ids = cosine_search(vec, embeddings, TOP_K_VEC)
    print(f"  向量搜尋初始 chunks：{seed_ids}")

    # Step 2: 找實體
    seed_entities = find_entities_in_chunks(seed_ids, graph)
    print(f"  找到實體：{', '.join(list(seed_entities)[:8])}" +
          ("..." if len(seed_entities) > 8 else ""))

    # Step 3: BFS 展開
    expanded = bfs_expand(seed_entities, graph, HOP_LIMIT)
    new_entities = expanded - seed_entities
    if new_entities:
        print(f"  圖遍歷新增實體：{', '.join(list(new_entities)[:6])}" +
              ("..." if len(new_entities) > 6 else ""))

    # Step 4: 收集所有相關 chunk IDs
    all_chunk_ids = set(seed_ids)
    for entity in expanded:
        if entity in graph["nodes"]:
            all_chunk_ids.update(graph["nodes"][entity]["chunk_ids"])

    # Step 5: 回傳 chunks（保持原始 chunk 順序）
    result = [chunks[i] for i in sorted(all_chunk_ids) if i < len(chunks)]
    # 用向量分數重新排序，取 Top-K
    if len(result) > TOP_K_VEC * 2:
        scored = []
        for c in result:
            cid = c["chunk_id"]
            emb_vec = embeddings[cid]
            score = float(np.dot(emb_vec / (np.linalg.norm(emb_vec) + 1e-10),
                                 vec / (np.linalg.norm(vec) + 1e-10)))
            scored.append((score, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        result = [c for _, c in scored[:TOP_K_VEC * 2]]

    return result

# ─── 問答 ─────────────────────────────────────────────────────────────────────

def answer(question: str, chunks: List[Dict], embeddings: np.ndarray, graph: Dict) -> None:
    print(f"\n搜尋相關資料（向量 + 圖遍歷）...")
    results = retrieve_graph(question, chunks, embeddings, graph)
    pages = sorted(set(c["page"] for c in results))
    print(f"參考頁碼：第 {', '.join(str(p) for p in pages)} 頁（共 {len(results)} 個片段）")

    context = "\n\n".join(f"[第{c['page']}頁]\n{c['text']}" for c in results)
    prompt = ANSWER_PROMPT.format(context=context, question=question)

    print(f"\nBRZ 專家：", end="", flush=True)
    stream_generate(prompt)
    print("─" * 50)

# ─── CLI ─────────────────────────────────────────────────────────────────────

def cmd_ingest(args):
    pdf_path = args.pdf
    if not Path(pdf_path).exists():
        print(f"錯誤：找不到 {pdf_path}"); sys.exit(1)

    print(f"[1/5] 解析 PDF: {pdf_path}")
    pages = extract_text_from_pdf(pdf_path)
    print(f"      共 {len(pages)} 頁")

    print("[2/5] 切割 chunks")
    ch = build_chunks(pages)
    print(f"      共 {len(ch)} 個 chunks")

    print(f"[3/5] 生成 embeddings")
    emb_list = []
    n = len(ch)
    for i, chunk in enumerate(ch):
        bar = "#" * int(30 * (i+1)/n) + "-" * (30 - int(30*(i+1)/n))
        print(f"\r  [{bar}] {i+1}/{n}", end="", flush=True)
        emb_list.append(embed_text(chunk["text"]))
    print()
    emb = np.array(emb_list, dtype=np.float32)

    print("[4/5] 抽取實體與關係，建立知識圖譜")
    graph = build_graph(ch)
    print(f"      節點數：{len(graph['nodes'])}，邊數：{len(graph['edges'])}")

    print("[5/5] 儲存索引")
    save_index(ch, emb, graph)
    print("完成！")


def cmd_chat(args):
    print("載入索引...", end=" ")
    chunks, embeddings, graph = load_index()
    print(f"OK（{len(chunks)} chunks，{len(graph['nodes'])} 個實體，{len(graph['edges'])} 條邊）")

    if args.query:
        answer(args.query, chunks, embeddings, graph)
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
            answer(q, chunks, embeddings, graph)


def cmd_show_graph(args):
    print("載入圖譜...", end=" ")
    _, _, graph = load_index()
    print("OK")

    nodes = graph["nodes"]
    edges = graph["edges"]

    print(f"\n=== 知識圖譜概覽 ===")
    print(f"節點數：{len(nodes)}，邊數：{len(edges)}")

    print(f"\n--- 高頻實體 Top-15 ---")
    sorted_nodes = sorted(nodes.items(), key=lambda x: x[1]["count"], reverse=True)
    for name, info in sorted_nodes[:15]:
        print(f"  {name}（出現 {info['count']} 次，關聯 {len(info['chunk_ids'])} 個片段）")

    print(f"\n--- 關係樣本（前 20 條）---")
    for edge in edges[:20]:
        print(f"  {edge['from']} --[{edge['rel']}]--> {edge['to']}")


def main():
    parser = argparse.ArgumentParser(description="Graph RAG — 知識圖譜增強取回")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="建立索引與知識圖譜")
    p_ingest.add_argument("pdf")
    p_ingest.set_defaults(func=cmd_ingest)

    p_chat = sub.add_parser("chat", help="問答")
    p_chat.add_argument("--query", "-q")
    p_chat.set_defaults(func=cmd_chat)

    p_show = sub.add_parser("show-graph", help="顯示知識圖譜")
    p_show.set_defaults(func=cmd_show_graph)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
