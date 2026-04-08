"""
05 — Agentic RAG（ReAct）
LLM 自主決定何時搜尋、搜尋什麼、何時回答。
不再是固定管線，而是動態的推理-行動迴圈。

工具：
  search(查詢)   → 在車型冊裡搜尋
  lookup(頁碼)   → 取得某頁完整內容
  finish(答案)   → 輸出最終答案

用法:
  python agentic_rag.py ingest ../brz_tw.pdf   # 建立索引（若未用 01 的）
  python agentic_rag.py chat                    # 互動模式
  python agentic_rag.py chat -q "問題"
  python agentic_rag.py chat -q "問題" --verbose   # 顯示完整推理過程
"""

import argparse
import io
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
SEARCH_TOP_K  = 3
MAX_STEPS     = 6   # ReAct 最大步數

SHARED_INDEX = Path(__file__).parent.parent / "01_naive_rag" / "index"
LOCAL_INDEX  = Path(__file__).parent / "index"

# ─── ReAct System Prompt ─────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
你是一個 RAG 代理人，負責回答關於 Subaru BRZ 台灣市場車型冊的問題。

你有以下工具可以使用：
- search(查詢文字): 在車型冊中搜尋相關內容，回傳最相關的片段
- lookup(頁碼): 取得指定頁面的完整文字內容（頁碼為整數）
- finish(答案): 當你有足夠資訊時，輸出最終答案給使用者

規則：
1. 每次只能呼叫一個工具
2. 必須嚴格遵守輸出格式
3. 如果搜尋結果不相關，試著換個角度重新搜尋
4. 最多 {max_steps} 步，超過就用現有資訊回答

輸出格式（每次必須包含這三行）：
思考：[你的推理過程]
工具：search 或 lookup 或 finish
輸入：[工具的輸入內容]"""

STEP_PROMPT = """\
{system}

[歷史記錄]
{history}
問題：{question}

請繼續（下一步）："""

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


def stream_generate(prompt: str) -> str:
    """串流生成並收集完整輸出。"""
    resp = requests.post(f"{OLLAMA_BASE}/api/generate",
        json={"model": GEN_MODEL, "prompt": prompt, "stream": True},
        stream=True, timeout=120)
    resp.raise_for_status()
    full = []
    for line in resp.iter_lines():
        if not line: continue
        data = json.loads(line)
        token = data.get("response", "")
        full.append(token)
        if data.get("done"): break
    return "".join(full)

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

# ─── 工具函式 ─────────────────────────────────────────────────────────────────

def tool_search(query: str, chunks: List[Dict], embeddings: np.ndarray) -> str:
    """向量搜尋，回傳格式化的結果字串。"""
    vec = np.array(embed_text(query), dtype=np.float32)
    q = vec / (np.linalg.norm(vec) + 1e-10)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    scores = (embeddings / norms) @ q
    top_idxs = list(np.argsort(scores)[::-1][:SEARCH_TOP_K])

    results = []
    for idx in top_idxs:
        c = chunks[idx]
        results.append(f"[第{c['page']}頁（相似度{scores[idx]:.2f}）]\n{c['text']}")
    return "\n\n".join(results)


def tool_lookup(page_num_str: str, chunks: List[Dict]) -> str:
    """取得指定頁面的所有 chunks，合併回傳。"""
    try:
        page_num = int(re.search(r"\d+", page_num_str).group())
    except Exception:
        return "錯誤：無法解析頁碼，請輸入整數。"

    page_chunks = [c for c in chunks if c["page"] == page_num]
    if not page_chunks:
        return f"找不到第 {page_num} 頁的內容（共 {max(c['page'] for c in chunks)} 頁）。"

    content = "\n\n".join(c["text"] for c in page_chunks)
    return f"[第 {page_num} 頁完整內容]\n{content}"

# ─── 解析 LLM 輸出 ───────────────────────────────────────────────────────────

def parse_action(text: str) -> Tuple[Optional[str], Optional[str], str]:
    """
    解析 LLM 輸出，尋找：
      思考：... / 工具：... / 輸入：...
    回傳 (tool_name, tool_input, thought)
    """
    thought = ""
    tool = None
    tool_input = None

    lines = text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("思考："):
            thought = line[3:].strip()
        elif line.startswith("工具："):
            tool = line[3:].strip().lower()
        elif line.startswith("輸入："):
            tool_input = line[3:].strip()

    # 容錯：若找不到標準格式，嘗試英文格式
    if not tool:
        for line in lines:
            if "search" in line.lower() and "：" in line:
                tool = "search"
                tool_input = line.split("：", 1)[-1].strip()
                break
            elif "lookup" in line.lower() and "：" in line:
                tool = "lookup"
                tool_input = line.split("：", 1)[-1].strip()
                break
            elif "finish" in line.lower() and "：" in line:
                tool = "finish"
                tool_input = line.split("：", 1)[-1].strip()
                break

    return tool, tool_input, thought

# ─── ReAct 主迴圈 ─────────────────────────────────────────────────────────────

def react_loop(question: str, chunks: List[Dict], embeddings: np.ndarray,
               verbose: bool = False) -> str:
    """
    執行 ReAct 迴圈，回傳最終答案。
    """
    history_lines = []
    system = SYSTEM_PROMPT.format(max_steps=MAX_STEPS)
    final_answer = None

    for step in range(MAX_STEPS):
        history = "\n".join(history_lines) if history_lines else "（尚無記錄）"
        prompt = STEP_PROMPT.format(system=system, history=history, question=question)

        raw = generate_text(prompt)

        if verbose:
            print(f"\n[步驟 {step+1}]\n{raw}")
        else:
            # 非 verbose 只顯示思考
            for line in raw.split("\n"):
                if line.strip().startswith("思考："):
                    print(f"  [{step+1}] {line.strip()}")
                    break

        tool, tool_input, thought = parse_action(raw)

        if not tool:
            # 解析失敗，嘗試直接視為 finish
            if len(raw.strip()) > 20:
                final_answer = raw.strip()
                break
            continue

        if tool == "finish":
            final_answer = tool_input or raw
            break

        elif tool == "search":
            observation = tool_search(tool_input, chunks, embeddings)
            history_lines.append(
                f"步驟{step+1}：search({tool_input})\n"
                f"觀察：{observation[:500]}{'...' if len(observation) > 500 else ''}"
            )

        elif tool == "lookup":
            observation = tool_lookup(tool_input, chunks)
            history_lines.append(
                f"步驟{step+1}：lookup({tool_input})\n"
                f"觀察：{observation[:500]}{'...' if len(observation) > 500 else ''}"
            )

        else:
            history_lines.append(f"步驟{step+1}：未知工具 {tool}")

    # 超過步數或沒有 finish，強制回答
    if final_answer is None:
        context = "\n\n".join(history_lines[-3:])  # 用最近的觀察
        fallback_prompt = (
            f"根據以下搜尋記錄，請用繁體中文回答問題。\n\n"
            f"{context}\n\n問題：{question}\n\n回答："
        )
        final_answer = generate_text(fallback_prompt)

    return final_answer.strip()

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
        print("找不到索引，請先執行: python agentic_rag.py ingest ../brz_tw.pdf")
        sys.exit(1)

    print(f"載入索引 ({index_dir.name})...", end=" ")
    chunks, embeddings = load_index(index_dir)
    print(f"OK（{len(chunks)} chunks）")
    print(f"模式：ReAct（最多 {MAX_STEPS} 步）\n")

    def run_question(q):
        print(f"\n思考中...")
        answer = react_loop(q, chunks, embeddings, verbose=args.verbose)
        print(f"\nBRZ 專家：{answer}")
        print("─" * 50)

    if args.query:
        run_question(args.query)
    else:
        print(f'互動問答模式（"exit" 離開）\n' + "─" * 50)
        while True:
            try:
                q = input("\n你：").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再見！"); break
            if not q: continue
            if q.lower() in ("exit", "quit", "bye", "掰掰"):
                print("再見！"); break
            run_question(q)


def main():
    parser = argparse.ArgumentParser(description="Agentic RAG — ReAct 推理-行動迴圈")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("pdf")
    p_ingest.set_defaults(func=cmd_ingest)

    p_chat = sub.add_parser("chat")
    p_chat.add_argument("--query", "-q")
    p_chat.add_argument("--verbose", "-v", action="store_true", help="顯示完整推理過程")
    p_chat.set_defaults(func=cmd_chat)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
