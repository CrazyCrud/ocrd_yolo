# OCR-D wrapper for yolo based on the ocrd_detectron2 wrapper

> YOLO-based region segmentation for OCR-D.
> You can find the ocrd_detectron2 wrapper here: https://github.com/bertsky/ocrd_detectron2/tree/master

This OCR-D processor uses YOLO models to detect and segment document regions in document images.

## Installation

```bash
pip install -e .
```

Or install via Dockerhub:
```commandline
- docker compose build
- docker-compose run ocrd-yolo
```
For CPU only:
```commandline
- docker compose build ocrd-yolo-cpu
- docker-compose run ocrd-yolo-cpu
```

## Usage

### Basic Usage

```bash
# Using a pre-trained model from resources
ocrd-yolo-segment \
    -I OCR-D-IMG \
    -O OCR-D-SEG-REGION \
    -p '{
        "model_weights": "yolo11s-example.pt",
        "categories": ["TextRegion:paragraph", "TextRegion:heading", "Border:page", "TableRegion", "ImageRegion"],
        "min_confidence": 0.5
    }'
```

### Table Segmentation

For segmenting table regions:

```bash
ocrd-yolo-segment \
    -I OCR-D-SEG-BLOCK \
    -O OCR-D-SEG-TABLE \
    -p '{
        "model_weights": "yolo11s-table.pt",
        "categories": ["TextRegion:columns", "TextRegion:header"],
        "level-of-operation": "table",
        "min_confidence": 0.7
    }'
```

## Available Models

- Models trained on historical, handwritten data can be for example downloaded here: [https://huggingface.co/collections/Riksarkivet/htrflow-v012-models](https://huggingface.co/collections/Riksarkivet/htrflow-v012-models) 

## Parameters

- `model_weights` (string, required): Path to YOLO model weights
- `categories` (array, required): Maps model classes to PAGE-XML region types
- `level-of-operation` (string, default: "page"): "page" or "table" level processing
- `min_confidence` (float, default: 0.5): Detection confidence threshold
- `postprocessing` (string, default: "full"): Post-processing mode
  - "full": NMS + morphological operations
  - "only-nms": Only non-maximum suppression
  - "only-morph": Only morphological operations
  - "none": No post-processing
- `debug_img` (boolean, default: false): Debug visualization
- `model_type` (string, default: "box"): Could be "box", "segmentation", and "obb"
- `device` (string, default: "cuda"): Computing device

## Category Mapping

Categories map model predictions to PAGE-XML region types:

```json
{
    "categories": [
        "TextRegion",           // Simple text region
        "TextRegion:heading",   // Text region with subtype
        "Border:page",          // Page border
        "ImageRegion",          // Image/figure
        "TableRegion",          // Table
        "GraphicRegion",        // Graphics/drawings
        "SeparatorRegion",      // Lines/separators
        "",                     // Skip this class
        "CustomRegion:formula"  // Custom region type
    ]
}
```

## Integration with OCR-D Workflow

### Complete OCR Workflow

```bash
# 1. Import images
ocrd-import ...

# 2. Binarize
ocrd-olena-binarize -I OCR-D-IMG -O OCR-D-BIN

# 3. Detect regions with YOLO
ocrd-yolo-segment -I OCR-D-BIN -O OCR-D-SEG-REGION \
    -p '{"model_weights": "yolo11s-example.pt", ...}'

# 4. Detect lines
ocrd-tesserocr-segment-line -I OCR-D-SEG-REGION -O OCR-D-SEG-LINE

# 5. OCR
ocrd-tesserocr-recognize -I OCR-D-SEG-LINE -O OCR-D-OCR
```

### Incremental Segmentation

The processor supports incremental segmentation - it won't overwrite existing regions:

```bash
# First pass: detect main regions
ocrd-yolo-segment -I OCR-D-IMG -O OCR-D-SEG-REGION-1 \
    -p '{"model_weights": "model1.pt", ...}'

# Second pass: detect additional regions
ocrd-yolo-segment -I OCR-D-SEG-REGION-1 -O OCR-D-SEG-REGION-2 \
    -p '{"model_weights": "model2.pt", ...}'
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or use a smaller models:

```bash
# Use smaller model
-p '{"model_weights": "yolo11n-example.pt"}'

# Or force CPU
-p '{"device": "cpu"}'
```

### Poor Detection Results

1. Check confidence threshold:
   ```bash
   -p '{"min_confidence": 0.3}'  # Lower threshold
   ```

2. Try different post-processing:
   ```bash
   -p '{"postprocessing": "only-nms"}'
   ```

3. Use appropriate model for document type

### Missing Regions

Enable debug visualization to see all detections:

```bash
-p '{"debug_img": true}'
```

## Remark
ChatGPT 4.5, 5 as well as Claude Opus 4.5 and Qwen 3.5 397B A17B have been used to generate this OCR-D extension.
At the beginning, the detectron2 extension was used as an example: github.com/bertsky/ocrd_detectron2
Generated code has been manually tested and iteratively improved using the listed models.
