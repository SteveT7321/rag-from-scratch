"""
04 — Multimodal RAG
用 Vision LLM (gemma4) 描述 PDF 裡的圖片，合併進向量索引。
解決 Naive RAG 對圖片、車色圖、規格截圖完全盲目的問題。

用法:
  python multimodal_rag.py ingest ../brz_tw.pdf    # 提取文字+圖片，建索引
  python multimodal_rag.py chat                     # 互動問答
  python multimodal_rag.py chat -q "BRZ有哪些顏色？"
  python multimodal_rag.py list-images              # 顯示提取的圖片清單
"""

import argparse
import base64
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
TOP_K         = 4
MIN_IMG_W     = 80    # 最小圖片寬度（像素），太小的跳過
MIN_IMG_H     = 80    # 最小圖片高度（像素）
MAX_IMG_DIM   = 1024  # 縮圖至最大邊長，節省 token

INDEX_DIR   = Path(__file__).parent / "index"
IMAGES_DIR  = Path(__file__).parent / "images"

IMAGE_DESCRIBE_PROMPT = """\
這是一張來自 Subaru BRZ 台灣市場車型冊的圖片。
請用繁體中文詳細描述這張圖片的內容，包括：
- 主要內容（車輛外觀、內裝、配件、顏色、文字等）
- 具體的顏色名稱或規格數值（若有）
- 圖片在車型冊中可能的用途

請直接描述，不要說「這張圖片顯示」或「圖中可以看到」等前綴。"""

ANSWER_PROMPT = """\
你是一位熟悉 Subaru BRZ 的專家。請根據以下資料回答問題。
資料可能包含文字片段和圖片描述。若資料中找不到答案，請直接說「資料中沒有相關資訊」。

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


def generate_text(prompt: str, images_b64: List[str] = None) -> str:
    payload = {"model": GEN_MODEL, "prompt": prompt, "stream": False}
    if images_b64:
        payload["images"] = images_b64
    resp = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=120)
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

# ─── PDF 文字提取 ─────────────────────────────────────────────────────────────

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


def build_text_chunks(pages: List[Dict]) -> List[Dict]:
    all_chunks, chunk_id = [], 0
    for p in pages:
        for t in split_text(p["text"]):
            all_chunks.append({
                "chunk_id": chunk_id, "page": p["page"],
                "text": t, "type": "text"
            })
            chunk_id += 1
    return all_chunks

# ─── PDF 圖片提取 ─────────────────────────────────────────────────────────────

def resize_image_bytes(img_bytes: bytes, max_dim: int) -> bytes:
    """將圖片縮小至 max_dim，節省 Vision LLM token 消耗。"""
    from PIL import Image
    img = Image.open(io.BytesIO(img_bytes))
    w, h = img.size
    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def extract_images_from_pdf(pdf_path: str) -> List[Dict]:
    """
    提取 PDF 中所有圖片，過濾太小的，回傳：
    {"page": int, "img_index": int, "width": int, "height": int, "bytes": bytes}
    """
    import fitz
    doc = fitz.open(pdf_path)
    images = []
    img_count = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        img_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(img_list):
            xref = img_info[0]
            try:
                pix = fitz.Pixmap(doc, xref)
            except Exception:
                continue

            w, h = pix.width, pix.height
            if w < MIN_IMG_W or h < MIN_IMG_H:
                continue  # 跳過太小的圖（icon、裝飾）

            # 轉成 PNG bytes
            if pix.n > 4:  # CMYK 轉 RGB
                pix = fitz.Pixmap(fitz.csRGB, pix)
            img_bytes = pix.tobytes("png")

            images.append({
                "page": page_num + 1,
                "img_index": img_count,
                "width": w,
                "height": h,
                "bytes": img_bytes,
            })
            img_count += 1

    doc.close()
    return images


def describe_image(img_bytes: bytes) -> str:
    """用 Vision LLM 描述圖片內容。"""
    try:
        resized = resize_image_bytes(img_bytes, MAX_IMG_DIM)
        img_b64 = base64.b64encode(resized).decode()
        description = generate_text(IMAGE_DESCRIBE_PROMPT, images_b64=[img_b64])
        return description.strip()
    except Exception as e:
        return f"（圖片描述失敗：{e}）"


def build_image_chunks(images: List[Dict], start_chunk_id: int) -> Tuple[List[Dict], List[Dict]]:
    """
    對每張圖片生成描述，建立 image chunks。
    同時把圖片存到 images/ 目錄。
    回傳 (image_chunks, image_metadata_list)
    """
    IMAGES_DIR.mkdir(exist_ok=True)
    chunks = []
    metadata = []
    chunk_id = start_chunk_id
    n = len(images)

    for i, img in enumerate(images):
        print(f"\r  描述圖片 {i+1}/{n}（第{img['page']}頁，{img['width']}×{img['height']}）... ",
              end="", flush=True)

        # 存圖片
        img_path = IMAGES_DIR / f"page{img['page']:02d}_img{img['img_index']:02d}.png"
        with open(img_path, "wb") as f:
            f.write(img["bytes"])

        # 生成描述
        description = describe_image(img["bytes"])
        print("OK")

        label = f"[圖片：第{img['page']}頁，{img['width']}×{img['height']}px]"
        full_text = f"{label}\n{description}"

        chunks.append({
            "chunk_id": chunk_id,
            "page": img["page"],
            "text": full_text,
            "type": "image",
            "img_path": str(img_path),
        })
        metadata.append({
            "chunk_id": chunk_id,
            "page": img["page"],
            "img_path": str(img_path),
            "width": img["width"],
            "height": img["height"],
            "description": description,
        })
        chunk_id += 1

    return chunks, metadata

# ─── 向量搜尋 ─────────────────────────────────────────────────────────────────

def cosine_search(query_vec: np.ndarray, embeddings: np.ndarray, k: int) -> List[int]:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    scores = (embeddings / norms) @ q
    return list(np.argsort(scores)[::-1][:k])

# ─── 索引 I/O ────────────────────────────────────────────────────────────────

def save_index(chunks, embeddings, img_metadata):
    INDEX_DIR.mkdir(exist_ok=True)
    with open(INDEX_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    np.save(str(INDEX_DIR / "embeddings.npy"), embeddings)
    with open(INDEX_DIR / "img_metadata.json", "w", encoding="utf-8") as f:
        json.dump(img_metadata, f, ensure_ascii=False, indent=2)


def load_index():
    for fname in ("chunks.pkl", "embeddings.npy"):
        if not (INDEX_DIR / fname).exists():
            print(f"錯誤：找不到 {INDEX_DIR / fname}，請先執行 ingest")
            sys.exit(1)
    with open(INDEX_DIR / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    embeddings = np.load(str(INDEX_DIR / "embeddings.npy"))
    return chunks, embeddings

# ─── 問答 ─────────────────────────────────────────────────────────────────────

def answer(question: str, chunks: List[Dict], embeddings: np.ndarray) -> None:
    print(f"\n搜尋相關資料（文字 + 圖片）...", end=" ", flush=True)
    vec = np.array(embed_text(question), dtype=np.float32)
    idxs = cosine_search(vec, embeddings, TOP_K)
    results = [chunks[i] for i in idxs]
    print(f"找到 {len(results)} 個片段")

    text_count = sum(1 for c in results if c.get("type") == "text")
    img_count  = sum(1 for c in results if c.get("type") == "image")
    print(f"  文字片段：{text_count}，圖片描述：{img_count}")

    pages = sorted(set(c["page"] for c in results))
    print(f"參考頁碼：第 {', '.join(str(p) for p in pages)} 頁")

    context = "\n\n".join(f"[第{c['page']}頁 {'(圖片)' if c.get('type')=='image' else ''}]\n{c['text']}"
                          for c in results)
    prompt = ANSWER_PROMPT.format(context=context, question=question)

    print(f"\nBRZ 專家：", end="", flush=True)
    stream_generate(prompt)
    print("─" * 50)

# ─── CLI ─────────────────────────────────────────────────────────────────────

def cmd_ingest(args):
    pdf_path = args.pdf
    if not Path(pdf_path).exists():
        print(f"錯誤：找不到 {pdf_path}"); sys.exit(1)

    print(f"[1/5] 解析 PDF 文字: {pdf_path}")
    pages = extract_text_from_pdf(pdf_path)
    print(f"      共 {len(pages)} 頁有文字")

    print("[2/5] 切割文字 chunks")
    text_chunks = build_text_chunks(pages)
    print(f"      共 {len(text_chunks)} 個文字 chunks")

    print("[3/5] 提取 PDF 圖片")
    images = extract_images_from_pdf(pdf_path)
    print(f"      共 {len(images)} 張有效圖片（過濾 < {MIN_IMG_W}×{MIN_IMG_H}px）")

    print("[4/5] Vision LLM 描述圖片")
    image_chunks, img_metadata = build_image_chunks(images, start_chunk_id=len(text_chunks))

    all_chunks = text_chunks + image_chunks
    print(f"      合計 {len(all_chunks)} 個 chunks（{len(text_chunks)} 文字 + {len(image_chunks)} 圖片）")

    print("[5/5] 生成 embeddings 並儲存")
    n = len(all_chunks)
    vectors = []
    for i, chunk in enumerate(all_chunks):
        bar = "#" * int(30*(i+1)/n) + "-" * (30 - int(30*(i+1)/n))
        print(f"\r  [{bar}] {i+1}/{n}", end="", flush=True)
        vectors.append(embed_text(chunk["text"]))
    print()
    embeddings = np.array(vectors, dtype=np.float32)

    save_index(all_chunks, embeddings, img_metadata)
    print("完成！")


def cmd_chat(args):
    print("載入索引...", end=" ")
    chunks, embeddings = load_index()
    text_n = sum(1 for c in chunks if c.get("type") == "text")
    img_n  = sum(1 for c in chunks if c.get("type") == "image")
    print(f"OK（{text_n} 文字 + {img_n} 圖片 chunks）")

    if args.query:
        answer(args.query, chunks, embeddings)
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
            answer(q, chunks, embeddings)


def cmd_list_images(args):
    meta_path = INDEX_DIR / "img_metadata.json"
    if not meta_path.exists():
        print("找不到圖片元數據，請先執行 ingest"); sys.exit(1)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"共 {len(metadata)} 張圖片：")
    for m in metadata:
        print(f"\n第{m['page']}頁 | {m['width']}×{m['height']}px | {m['img_path']}")
        print(f"  描述：{m['description'][:120]}...")


def main():
    parser = argparse.ArgumentParser(description="Multimodal RAG — 文字 + 圖片")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="提取文字+圖片，建立索引")
    p_ingest.add_argument("pdf")
    p_ingest.set_defaults(func=cmd_ingest)

    p_chat = sub.add_parser("chat", help="問答")
    p_chat.add_argument("--query", "-q")
    p_chat.set_defaults(func=cmd_chat)

    p_list = sub.add_parser("list-images", help="列出提取的圖片")
    p_list.set_defaults(func=cmd_list_images)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
