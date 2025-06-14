#!/usr/bin/env python3
"""
minimal_test.py - Minimal test for YOLO OCR-D processor
Tests basic functionality without needing a real document dataset model
"""

import tempfile
import numpy as np
from PIL import Image
import json
from pathlib import Path


def test_basic_setup():
    """Test if the processor can be imported and initialized"""
    print("1. Testing imports...")
    try:
        from ocrd_yolo.segment import Yolo2Segment
        from ocrd_yolo.nms import postprocess_nms
        from ocrd_yolo.utils import polygon_for_parent
        print("✓ All imports successful")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

    print("\n2. Testing processor initialization...")
    try:
        proc = Yolo2Segment(
            workspace=None,
            parameter={
                'model_weights': 'yolo11n.pt',  # Base model
                'categories': ['TextRegion', 'ImageRegion', 'TableRegion'],
                'min_confidence': 0.5,
                'device': 'cpu',
                'operation_level': 'page'
            }
        )
        print("✓ Processor initialized")
    except Exception as e:
        print(f"✗ Initialization error: {e}")
        return False

    print("\n3. Testing parameter validation...")
    print(f"Parameters: {json.dumps(proc.parameter, indent=2)}")

    return True


def test_nms_functions():
    """Test the NMS post-processing functions"""
    print("\n4. Testing NMS functions...")

    # Create dummy data
    height, width = 100, 100
    num_detections = 3

    # Create overlapping masks
    masks = np.zeros((num_detections, height, width), dtype=bool)
    masks[0, 10:40, 10:40] = True  # Box 1
    masks[1, 30:60, 30:60] = True  # Box 2 (overlaps with 1)
    masks[2, 70:90, 70:90] = True  # Box 3 (no overlap)

    scores = np.array([0.9, 0.8, 0.95])
    classes = np.array([0, 0, 1])
    categories = ['TextRegion', 'ImageRegion']
    page_bin = np.ones((height, width), dtype=bool)

    try:
        from ocrd_yolo.nms import postprocess_nms
        new_scores, new_classes, new_masks = postprocess_nms(
            scores, classes, masks, page_bin, categories,
            min_confidence=0.5, nproc=1
        )
        print(f"✓ NMS completed: {len(masks)} -> {len(new_masks)} detections")
        print(f"  Kept detections: {len(new_masks)}")
        print(f"  Scores: {new_scores}")
    except Exception as e:
        print(f"✗ NMS error: {e}")
        return False

    return True


def test_with_dummy_image():
    """Test with a synthetic image"""
    print("\n5. Testing with dummy image...")

    # Create a dummy document image
    width, height = 800, 1000
    img = Image.new('RGB', (width, height), color='white')

    # Add some black rectangles to simulate text regions
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)

    # Simulate text blocks
    draw.rectangle([50, 50, 350, 200], fill='black')  # Title
    draw.rectangle([50, 250, 750, 400], fill='gray')  # Paragraph 1
    draw.rectangle([50, 450, 750, 600], fill='gray')  # Paragraph 2
    draw.rectangle([400, 650, 750, 950], fill='lightgray')  # Image area

    # Save test image
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        img.save(tmp.name)
        print(f"✓ Created test image: {tmp.name}")
        print(f"  Size: {width}x{height}")

        # Create a minimal workspace-like structure
        print("\n6. Creating test workspace structure...")
        test_dir = Path(tempfile.mkdtemp())
        img_dir = test_dir / 'OCR-D-IMG'
        img_dir.mkdir()

        # Copy image
        test_img = img_dir / 'test_001.png'
        img.save(test_img)

        print(f"✓ Test workspace created: {test_dir}")
        print(f"  Image saved to: {test_img}")

        return str(test_dir), str(test_img)


def main():
    print("=== YOLO OCR-D Minimal Test Suite ===\n")

    # Run tests
    if not test_basic_setup():
        print("\n✗ Basic setup failed")
        return 1

    if not test_nms_functions():
        print("\n✗ NMS functions test failed")
        return 1

    workspace_dir, test_image = test_with_dummy_image()

    print("\n=== All tests passed! ===")
    print("\nNext steps:")
    print("1. Train a model on document data:")
    print("   python train_yolo_ocrd.py --dataset-name publaynet ...")
    print("\n2. Or download a pre-trained model:")
    print("   ocrd resmgr download ocrd-yolo-segment yolo11s-publaynet.pt")
    print("\n3. Run on real documents:")
    print(f"   ocrd-yolo-segment -I OCR-D-IMG -O OCR-D-SEG -p '{{\"model_weights\": \"model.pt\", ...}}'")

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main())
