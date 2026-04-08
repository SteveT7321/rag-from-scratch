# 04 — Multimodal RAG

## 狀態：待實作

## 核心問題

BRZ 車型冊裡有大量圖片：車色展示、內裝照片、規格表格截圖。

Naive RAG 直接丟掉這些，導致：
- 問「有哪些顏色？」→ 回答「資料中沒有相關資訊」
- 問「儀表板長什麼樣？」→ 無法描述

## 解法架構

### 方法一：圖片轉文字（Image Captioning）

把 PDF 裡的圖片用 Vision LLM 描述成文字，再一起 embed 進向量庫。

```
PDF
  │
  ├─ 文字頁 → 原來的文字提取流程
  │
  └─ 圖片   → [Vision LLM] → 圖片描述文字
                               "這張圖展示了 BRZ 的 Crystal White Pearl 車色..."
                                     │
                                     ▼
                               一起 embed 進向量庫
```

**Vision LLM 選擇**：`gemma4:e4b` 本身支援多模態，直接用！

### 方法二：表格結構化提取

PDF 的表格（規格表）有時會被 PyMuPDF 提取成凌亂的文字流，需特別處理：

```python
# PyMuPDF 可以提取表格 bounding box
page.find_tables()  # 回傳 TableFinder 物件
```

把表格轉成 Markdown 格式，語意更清晰：

```
原始（凌亂）：
"排氣量 2,387 c.c. 最大馬力 234 ps/7,000rpm 最大扭力"

轉換後（Markdown 表格）：
| 項目 | 數值 |
|------|------|
| 排氣量 | 2,387 c.c. |
| 最大馬力 | 234 ps / 7,000 rpm |
```

## 實作計畫

### 圖片提取（PyMuPDF）

```python
doc = fitz.open("brz_tw.pdf")
for page_num, page in enumerate(doc):
    # 提取頁面上的所有圖片
    image_list = page.get_images(full=True)
    for img_index, img in enumerate(image_list):
        xref = img[0]
        pix = fitz.Pixmap(doc, xref)
        # 存成 PNG，再送給 Vision LLM
        pix.save(f"images/page{page_num+1}_img{img_index}.png")
```

### 圖片描述（Vision LLM）

```python
import base64

def describe_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "gemma4:e4b",
            "prompt": "請用繁體中文詳細描述這張圖片的內容，包括顏色、物件、文字等細節。",
            "images": [img_b64],
            "stream": False,
        }
    )
    return resp.json()["response"]
```

### 合併索引

```python
# 文字 chunk + 圖片描述 chunk 一起 embed
all_chunks = text_chunks + image_description_chunks
embeddings = embed_batch(all_chunks)
```

每個 image chunk 的 metadata：

```python
{
    "chunk_id": 42,
    "type": "image",          # 新增 type 欄位
    "page": 4,
    "image_path": "images/page4_img0.png",
    "text": "這張圖展示了 BRZ 的車色選擇，包括..."
}
```

### 檔案結構

```
04_multimodal_rag/
├── plan.md
├── requirements.txt
├── extract_images.py    # PDF 圖片提取
├── caption.py           # Vision LLM 圖片描述
├── extract_tables.py    # 表格結構化
├── multimodal_rag.py    # 整合 CLI
├── images/              # 提取的圖片暫存
└── index/
    ├── chunks.pkl        # 含 text + image 兩種 chunk
    └── embeddings.npy
```

### 建議套件

```
pymupdf==1.23.8
numpy
requests
Pillow              # 圖片格式處理（PNG/JPEG 轉換）
```

## 預期效果

| 問題 | Naive RAG | Multimodal RAG |
|------|-----------|----------------|
| 「BRZ 有哪些顏色？」 | 資料中沒有相關資訊 | Crystal White Pearl、Sapphire Blue... |
| 「儀表板有什麼特色？」 | 部分 | 描述圖片中的 7 吋 TFT 顯示器 |
| 「規格表的引擎排氣量？」 | 可能取到凌亂文字 | 從結構化表格精確取出 |

## 挑戰

- 圖片描述需要 Vision LLM，速度較慢（每張圖約 5-15 秒）
- BRZ 車型冊有車色色塊圖，LLM 可能描述成「一個色塊，顏色為...」
- 圖片 token 消耗大，需控制解析度
