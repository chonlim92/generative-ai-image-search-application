# Implementation Details

**Author: Chong Kiat Lim**

---

## Overview

This project implements a **Generative AI Natural Language Image Mining** system that enables users to search through a collection of images using natural language prompts. It leverages multi-modal AI models to compute semantic similarity between text descriptions and images.

![Application GUI](images/example_gui.jpg)

---

## Architecture

The application consists of two variants:

### 1. CLIP-Only Version (`GenAI_Image_Search_application.py`)

Uses OpenAI's CLIP (Contrastive Language-Image Pre-training) model to directly compute text-image similarity via softmax-based probability ranking.

**Pipeline:**
1. User uploads images and provides a natural language prompt
2. Images are preprocessed (converted to RGB)
3. CLIP processes both text and images jointly
4. `logits_per_image` is computed and softmax is applied across images
5. Results are sorted by probability and filtered against the user-defined threshold

### 2. CLIP + YOLOv5 Version (`GenAI_Image_Search_application_YOLOv5.py`)

Combines YOLOv5 object detection with CLIP embeddings for enhanced search accuracy using cosine similarity.

**Pipeline:**
1. User uploads images and provides a natural language prompt
2. YOLOv5 performs object detection on each image
3. CLIP generates normalized embeddings for both the text prompt and each image independently
4. Cosine similarity is calculated between text and image embeddings
5. Results are sorted by similarity score and filtered by threshold

---

## Variant Comparison

| Aspect | CLIP-Only (`GenAI_Image_Search_application.py`) | CLIP + YOLOv5 (`GenAI_Image_Search_application_YOLOv5.py`) |
|--------|------------------------------------------------|-----------------------------------------------------------|
| **Models** | CLIP only | CLIP + YOLOv5 |
| **Similarity method** | Joint text-image processing → `logits_per_image` → softmax across all images | Separate embeddings → cosine similarity per image |
| **Scoring** | Softmax probabilities (relative — all images sum to ~100%) | Cosine similarity (absolute — each image scored independently) |
| **Object detection** | None | YOLOv5 detects objects in each image before CLIP comparison |
| **UI** | Full Gradio Blocks (gallery, thumbnails, CSV export, file upload) | Simple `gr.Interface` (basic inputs/outputs) |
| **Features** | Thumbnails, CSV export, probability panel, image selection events | Minimal — gallery + text output only |
| **Embedding approach** | CLIP processes text + images together in one forward pass | CLIP extracts text features and image features separately, then computes cosine distance |

**Key practical difference**: The CLIP-only version gives *relative* rankings (probabilities redistribute if you add/remove images). The YOLOv5 version gives *absolute* similarity scores per image, making it more stable for threshold-based filtering regardless of batch size.

---

## Key Components

### Image Preprocessing

```python
def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image
```

Ensures all images are in RGB format regardless of source format (RGBA, grayscale, palette, CMYK).

### Similarity Computation

| Version | Method | Metric |
|---------|--------|--------|
| CLIP-Only | Joint text-image processing | Softmax over logits_per_image |
| CLIP + YOLOv5 | Separate embedding extraction | Cosine similarity |

### Result Export

Results are exported as semicolon-delimited CSV files with columns: `Index`, `Image File Name`, `Probability`.

---

## Model Configuration

Models are configured via environment variables in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `CLIP_MODEL_NAME` | `openai/clip-vit-base-patch32` | Hugging Face CLIP model identifier |
| `YOLO_MODEL_NAME` | `yolov5s` | YOLOv5 model variant (s/m/l/x) |

---

## GPU Acceleration

The application automatically detects CUDA availability:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

All model inference and tensor operations are performed on the detected device.

---

## UI Framework

The application uses **Gradio Blocks** for the web interface with:
- Dark theme enforcement via JavaScript
- File upload with image type filtering
- Interactive gallery with selection events
- Real-time probability display
- CSV export and download functionality

---

## Testing

Unit tests use `pytest` with mocked ML models to ensure fast execution without GPU or model weights:

```bash
pytest tests/ -v
```

Tests cover:
- Image preprocessing (format conversion, dimension preservation)
- Base64 encoding/decoding
- CSV export formatting and content
- Thumbnail generation
- Search pipeline with mocked model outputs
- Edge cases (empty inputs, invalid files)
