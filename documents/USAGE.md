# Usage Guide

**Author: Chong Kiat Lim**

---

## Prerequisites

- Python 3.9+
- CUDA-compatible GPU (optional, for faster inference)

---

## Installation

### 1. Create and activate a virtual environment

```bash
python -m venv .genai_imagesearch_venv

# Windows
.\.genai_imagesearch_venv\Scripts\activate

# Linux/macOS
source .genai_imagesearch_venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables (optional)

Copy or edit the `.env` file to customize model names:

```env
CLIP_MODEL_NAME=openai/clip-vit-base-patch32
YOLO_MODEL_NAME=yolov5s
```

If not set, the defaults above will be used.

---

## Running the Application

### CLIP-Only Version

```bash
python GenAI_Image_Search_application.py
```

### CLIP + YOLOv5 Version

```bash
python GenAI_Image_Search_application_YOLOv5.py
```

Both versions will launch a Gradio web interface with a shareable public link.

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

## Application Workflow

![Application GUI](images/example_gui.jpg)

1. **Upload Images** — Click the "Upload Image Files" area to select one or more image files. Supported formats: JPG, PNG, BMP, TIFF.

2. **Enter Search Prompt** — Type a natural language description in the "Prompt for Image Mining" text box. Example: `"low road curb with pedestrian walkway at road junction"`

3. **Set Probability Threshold** — Drag the slider to set a minimum confidence percentage (1–100%). Images below this threshold are excluded from results.

4. **Start Mining** — Click the "Start Image Mining" button. Processing time depends on the number of images and GPU availability.

5. **View Results** — Matched images appear in the "Mining Results" gallery, sorted by relevance. Click any image to view it full-size.

6. **Review Probabilities** — The probability panel shows confidence scores. Selecting a single image shows its individual score; deselecting shows all scores.

7. **Export to CSV** — Click "Export Result as CSV" to generate a downloadable CSV file with columns: Index, Image File Name, Probability.

---

## Running Tests

```bash
pytest tests/ -v
```

Tests run with mocked ML models and do not require GPU or downloaded weights.

---

## Tips

- For best results, use descriptive prompts that match visual content (e.g., objects, scenes, colors).
- The YOLOv5 version provides object-aware search and may be more accurate for object-specific queries.
- Higher thresholds reduce false positives but may miss borderline matches.
- GPU acceleration significantly reduces processing time for large image batches.

---

## Disclaimer

The mining results might still contain **False Positives** and **False Negatives**. Manual review of the results is strongly recommended.
