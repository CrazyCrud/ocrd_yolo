# @ai-generated model="gpt-4.5,claude opus 4.5,qwen 3.5 397B A17B,"


from __future__ import absolute_import

import cv2
import numpy as np
from PIL import Image
import torch
from typing import Optional

from ultralytics import YOLO
from shapely.geometry import Polygon

from ocrd_utils import (
    coordinates_of_segment,
    coordinates_for_segment,
    points_from_polygon,
)
from ocrd_models.ocrd_page import (
    OcrdPage,
    PageType,
    BorderType,
    AdvertRegionType,
    ChartRegionType,
    ChemRegionType,
    CustomRegionType,
    GraphicRegionType,
    ImageRegionType,
    LineDrawingRegionType,
    MapRegionType,
    MathsRegionType,
    MusicRegionType,
    NoiseRegionType,
    SeparatorRegionType,
    TableRegionType,
    TextRegionType,
    TextLineType,
    UnknownRegionType,
    CoordsType,
    AlternativeImageType
)
from ocrd_models.ocrd_page_generateds import (
    ChartTypeSimpleType,
    GraphicsTypeSimpleType,
    TextTypeSimpleType
)
from ocrd import Processor, OcrdPageResult, OcrdPageResultImage

from .nms import postprocess_nms, postprocess_morph
from .utils import polygon_for_parent, _ensure_consistent_crops


class Yolo2Segment(Processor):
    max_workers = 1

    @property
    def executable(self):
        return 'ocrd-yolo-segment'

    def setup(self):
        if self.parameter['device'] == 'cpu' or not torch.cuda.is_available():
            device = "cpu"
        else:
            device = self.parameter['device']
        self.logger.info(f"Using device {device}")

        # Load model
        model_weights = self.parameter['model_weights']

        # Try to resolve as resource first
        try:
            model_weights = self.resolve_resource(model_weights)
        except Exception:
            # If not a resource, check if it's a valid file path
            import os
            if not os.path.exists(model_weights):
                raise FileNotFoundError(f"Model file not found: {model_weights}")

        self.logger.info(f"Loading YOLO weights from {model_weights}")
        self.model = YOLO(model_weights)
        self.model.to(device)

        # Get parameters
        self.min_confidence = float(self.parameter.get('min_confidence', 0.5))
        self.categories = self.parameter['categories']
        self.postprocessing = self.parameter['postprocessing']

        # Validate categories match model classes
        model_classes = self.model.model.names if hasattr(self.model.model, 'names') else {}
        self.logger.info(f"Model has {len(model_classes)} classes")

    def process_page_pcgts(self, *input_pcgts: Optional[OcrdPage], page_id: Optional[str] = None) -> OcrdPageResult:
        """
        Use YOLO to segment each page.

        `level-of-operation` controls where we operate:
          - "page":   on the full page
          - "table":  inside existing TableRegions
          - "region": inside existing TextRegions
        """

        pcgts = input_pcgts[0]
        result = OcrdPageResult(pcgts)

        level = self.parameter.get('level-of-operation')

        page = pcgts.get_Page()
        page_image_raw, page_coords, page_image_info = self.workspace.image_from_page(
            page, page_id, feature_filter='raw')

        # For morphological post-processing, a binarized image is needed
        if self.postprocessing != 'none':
            try:
                page_image_bin, _, _ = self.workspace.image_from_page(
                    page, page_id, feature_selector='binarized')
                page_image_raw, page_image_bin = _ensure_consistent_crops(
                    page_image_raw, page_image_bin)
            except Exception:
                self.logger.warning("No binarized image found, creating from raw image")
                page_image_bin = page_image_raw.convert('L').point(
                    lambda x: 0 if x < 128 else 255, '1'
                )
        else:
            page_image_bin = page_image_raw

        # Determine zoom level
        if page_image_info.resolution != 1:
            dpi = page_image_info.resolution
            if page_image_info.resolutionUnit == 'cm':
                dpi = round(dpi * 2.54)
            zoom = 300.0 / dpi
        else:
            dpi = None
            zoom = 1.0

        resize_mode = self.parameter.get('resize_mode', 'none')

        if resize_mode == 'none':
            zoomed = 1.0
        elif resize_mode == 'auto':
            # Original behavior
            if zoom < 2.0:
                zoomed = zoom / 2.0
            else:
                zoomed = 1.0
        elif resize_mode == 'fixed':
            # Resize to specific size
            target_size = self.parameter.get('target_size', 1024)
            zoomed = target_size / max(page_image_raw.width, page_image_raw.height)
        else:
            self.logger.warning(f"Unknown resize_mode {resize_mode}, falling back to 'none'")
            zoomed = 1.0

        if level == 'page':
            segments = [page]
        elif level == 'table':
            segments = page.get_AllRegions(depth=1, classes=['Table'])
            if not segments:
                self.logger.warning(f"No existing TableRegions on page {page_id}")
        elif level in ('region', 'text-region'):
            # TextRegions
            segments = page.get_AllRegions(depth=1, classes=['Text'])
            if not segments:
                self.logger.warning(f"No existing TextRegions on page {page_id}")
        else:
            raise ValueError(f"Unknown level-of-operation / operation_level: {level}")

        for segment in segments:
            # Get existing regions for NMS or masking
            def at_segment(region):
                return region.parent_object_ is segment

            regions = list(filter(at_segment, page.get_AllRegions()))

            if isinstance(segment, PageType):
                image_raw = page_image_raw
                image_bin = page_image_bin
                coords = page_coords
            else:
                image_raw, coords = self.workspace.image_from_segment(
                    segment, page_image_raw, page_coords, feature_filter='raw')
                if self.postprocessing != 'none':
                    try:
                        image_bin, _ = self.workspace.image_from_segment(
                            segment, page_image_bin, page_coords)
                        image_raw, image_bin = _ensure_consistent_crops(
                            image_raw, image_bin)
                    except Exception:
                        # Create binarized from raw if not available
                        image_bin = image_raw.convert('L').point(
                            lambda x: 0 if x < 128 else 255, '1'
                        )
                else:
                    image_bin = image_raw

            # Ensure RGB or binary formats
            if image_raw.mode == '1':
                image_raw = image_raw.convert('L')
            image_raw = image_raw.convert(mode='RGB')
            image_bin = image_bin.convert(mode='1')

            # Reduce resolution if needed
            if zoomed != 1.0:
                image_bin = image_bin.resize(
                    (int(image_raw.width * zoomed),
                     int(image_raw.height * zoomed)),
                    resample=Image.Resampling.BICUBIC)
                image_raw = image_raw.resize(
                    (int(image_raw.width * zoomed),
                     int(image_raw.height * zoomed)),
                    resample=Image.Resampling.BICUBIC)

            # Convert to numpy arrays
            array_raw = np.array(image_raw)
            array_bin = np.array(image_bin)
            array_bin = ~array_bin  # Invert for processing

            # Pass `level` through so _process_segment can distinguish page/table/region behaviour
            image = self._process_segment(
                segment, regions, coords,
                array_raw, array_bin, zoomed,
                page_id, level
            )
            if image:
                result.images.append(image)

        return result

    def _process_segment(self, segment, ignore, coords, array_raw, array_bin, zoomed, page_id, level) -> Optional[
        OcrdPageResultImage]:
        segtype = segment.__class__.__name__[:-4]
        segment.set_custom('coords=%s' % coords['transform'])
        height, width = array_raw.shape[:2]

        # Estimate scale for morphological operations
        scale = 43
        if self.postprocessing in ['full', 'only-morph']:
            _, components = cv2.connectedComponents(array_bin.astype(np.uint8))
            _, counts = np.unique(components, return_counts=True)
            if counts.shape[0] > 1:
                counts = np.sqrt(3 * counts)
                counts = counts[(5 < counts) & (counts < 100)]
                scale = int(np.median(counts))
                self.logger.debug(f"estimated scale: {scale}")

        self.logger.info(f"Feeding YOLO: array_raw shape={array_raw.shape}, dtype={array_raw.dtype}")

        use_end2end = False
        if hasattr(self.model.model, "end2end"):
            use_end2end = bool(self.model.model.end2end)

        # Run YOLO inference and set end2end to false if using older YOLO models
        pil = Image.fromarray(array_raw)

        if use_end2end:
            self.logger.info("Use YOLO end2end")
        else:
            self.logger.info("Use YOLO without end2end (old behaviour)")
        
        results = self.model(pil, conf=self.min_confidence, verbose=False, end2end=use_end2end)

        n_boxes = len(results[0].boxes or [])
        n_masks = len(getattr(results[0], 'masks', []) or [])
        self.logger.info(f"Wrapper: YOLO returned {n_boxes} boxes and {n_masks} masks")

        if not results or not results[0].boxes:
            self.logger.warning(f"Detected no regions on {segtype} {segment.id}")
            return None
        else:
            self.logger.info(f"YOLO inference complete: {results}")
            self.logger.info(f"Raw detections: {len(results[0].boxes)}")
            for i, box in enumerate(results[0].boxes):
                cls = int(box.cls)
                conf = float(box.conf)
                self.logger.info(
                    f" Detection {i}: class={cls} ({self.categories[cls] if cls < len(self.categories) else 'unknown'}), conf={conf:.3f}"
                )

        # Extract detections from YOLO results
        result = results[0]
        boxes = result.boxes

        # Get the masks or boxes if no mask have been found
        use_boxes_as_masks = self.parameter.get('use_boxes_as_masks', True)
        masks = None
        # Get masks if boxes shouldn't be used
        if hasattr(result, 'masks') and result.masks is not None and not use_boxes_as_masks:
            masks = result.masks.data.cpu().numpy()
            
        if use_boxes_as_masks or masks is None:
            self.logger.info("Using bounding boxes to create masks")
            masks = []
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                mask = np.zeros((height, width), dtype=np.uint8)
                x1, x2 = int(x1 * width / result.orig_shape[1]), int(x2 * width / result.orig_shape[1])
                y1, y2 = int(y1 * height / result.orig_shape[0]), int(y2 * height / result.orig_shape[0])
                # Add static margin to the boxes
                margin = 3
                x1, y1, x2, y2 = max(0, x1 - margin), max(0, y1 - margin), min(width, x2 + margin), min(height, y2 + margin)
                mask[y1:y2, x1:x2] = 1
                masks.append(mask > 0)
            masks = np.array(masks)
        else:
            # Take masks and resize them based on image width and height, but only when the zoom is not euqal to 1
            if masks.shape[1:] != (height, width):
                masks_resized = []
                for mask in masks:
                    mask_resized = cv2.resize(mask.astype(np.uint8), (width, height),
                                                interpolation=cv2.INTER_NEAREST)
                    masks_resized.append(mask_resized > 0.5)
                masks = np.array(masks_resized)

        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        # Filter by categories if specified
        if not all(self.categories):
            keep_indices = [i for i, cls in enumerate(classes)
                            if cls < len(self.categories) and self.categories[cls]]
            if not keep_indices:
                self.logger.warning(f"No detections for selected categories on {segtype} {segment.id}")
                return None
            masks = masks[keep_indices]
            scores = scores[keep_indices]
            classes = classes[keep_indices]

        # Handle existing regions for NMS and prepare the post-processing steps
        if len(ignore) and not isinstance(segment, PageType):
            scores = np.insert(scores, 0, 1.0, axis=0)
            classes = np.insert(classes, 0, -1, axis=0)
            masks = np.insert(masks, 0, 0, axis=0)
            mask0 = np.zeros(masks.shape[1:], np.uint8)
            for i, region in enumerate(ignore):
                polygon = coordinates_of_segment(region, None, coords)
                if zoomed != 1.0:
                    polygon = np.round(polygon * zoomed).astype(int)
                cv2.fillPoly(mask0, pts=[polygon], color=(255,))
            if np.count_nonzero(mask0):
                masks[0] = mask0 > 0

        # Apply post-processing on the mask detection
        if self.postprocessing in ['full', 'only-nms']:
            scores, classes, masks = postprocess_nms(
                scores, classes, masks, array_bin, self.categories,
                min_confidence=self.min_confidence, nproc=8, logger=self.logger)

        if self.postprocessing in ['full', 'only-morph']:
            _, components = cv2.connectedComponents(array_bin.astype(np.uint8))
            scores, classes, masks = postprocess_morph(
                scores, classes, masks, components, nproc=8, logger=self.logger)

        # Remove placeholder for existing regions due to NMS step above
        if len(ignore):
            scores = scores[1:]
            classes = classes[1:]
            masks = masks[1:]

        detect_page_border = True
        # Convert masks to regions or lines
        region_no = 0
        line_no = 0

        self.logger.info(f"Starting main loop with {len(masks)} masks, {len(classes)} classes, {len(scores)} scores")

        for idx, (mask, class_id, score) in enumerate(zip(masks, classes, scores)):
            if class_id >= len(self.categories):
                self.logger.warning(f"Class id {class_id} out of range for categories (len={len(self.categories)}) on segment {segtype} {segment.id}")
                continue

            category = self.categories[class_id]
            self.logger.info(f"=== Loop iteration {idx + 1}/{len(masks)}: {category} (class={class_id}, score={score}) ===")

            if not category:
                self.logger.warning(f"Category is empty/None for class {class_id}")
                continue

            self.logger.info(f"Processing non-border region: {category}")

            self.logger.info(f"YOLO orig_shape: {result.orig_shape}, mask xy shape: {result.masks.xy[idx].shape}")
            self.logger.info(f"Image shape after resize: {array_raw.shape[:2]}")

            # Special handling for page class
            if category.startswith('Border') and isinstance(segment, PageType):
                if not detect_page_border or level != 'page':
                    self.logger.info("Skipping page border detection (disabled in config)")
                    continue
                # Check if Border already exists
                if segment.get_Border() is not None:
                    self.logger.warning(f"Page already has a Border, skipping new border with score {score}")
                    continue
                self.logger.info(f"Processing page boundary with score {score}")

                # Newer YOLO models should already return closed contours
                if idx < len(result.masks.xy) and not use_boxes_as_masks:
                    border_contour_xy = result.masks.xy[idx]
                    page_polygon = border_contour_xy.astype(np.float32)
                else:
                    mask_uint8 = mask.astype(np.uint8)
                    border_kernel_size = max(10, scale // 2)
                    kernel = np.ones((border_kernel_size, border_kernel_size), np.uint8)
                    mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
                    
                    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not contours:
                        self.logger.warning("Could not extract page boundary contour")
                        continue

                    all_points = np.concatenate(contours)
                    page_polygon = cv2.convexHull(all_points)[:, 0, :]

                page_polygon = coordinates_for_segment(page_polygon, None, coords)
                if page_polygon.shape[0] < 3:
                    self.logger.warning("Border polygon has <3 points")
                    continue

                if zoomed != 1.0:
                    page_polygon = page_polygon / zoomed

                # Create Border element
                border_coords = CoordsType(points_from_polygon(page_polygon), conf=score)
                border = BorderType(Coords=border_coords)
                segment.set_Border(border)
                self.logger.info(f"Set page Border from 'page' detection with conf {score}")

                # Skip creating a region for this
                continue
            
            # Create contours
            # Newer YOLO models should already return closed contours
            if idx < len(result.masks.xy) and not use_boxes_as_masks:
                raw_contour = result.masks.xy[idx].astype(np.float32)   # (P, 2)
                source = "model polygon"
            else:
                # fallback – old contour extraction (kept only for legacy models)
                mask_uint8 = mask.astype(np.uint8)
                kernel_size = max(3, min(scale // 5, 15))
                if kernel_size % 2 == 0:
                    kernel_size += 1
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
                mask = mask_closed > 0

                invalid = True
                contours = []
                for _ in range(10):
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if len(contours) == 1 and len(contours[0]) > 3:
                        invalid = False
                        break
                    mask = cv2.dilate(mask.astype(np.uint8), np.ones((scale, scale), np.uint8)) > 0

                if invalid:
                    self.logger.warning(f"Ignoring non-contiguous {len(contours)} region for {category}")
                    continue
                raw_contour = contours[0][:, 0, :].astype(np.float32)
                source = "fallback contour"

            if self.parameter.get('debug_img') != 'none':
                vis_img = array_raw.copy()
                pts = raw_contour.astype(np.int32)
                cv2.polylines(vis_img, [pts], isClosed=True, color=(0,255,0), thickness=2)
                cv2.imwrite(f"/tmp/debug_polygon_{segment.id}_{idx}.png", vis_img)

            if zoomed != 1.0:
                raw_contour = raw_contour / zoomed

            # Map into page coords
            # Using Qwen 3.5 397B A17B, it was discovered that polygon_for_parent removes too many points from the polygons.
            page_poly = coordinates_for_segment(raw_contour, None, coords)
            if level != 'page':
                page_poly = polygon_for_parent(page_poly, segment)
                if page_poly is None:
                    self.logger.warning(f"Ignoring clipped-away region for {category}")
                    continue
                self.logger.info(f"After polygon_for_parent: {len(page_poly)} points")
            else:
                self.logger.info("Skipping polygon_for_parent (page level - coords already absolute)")
                page_poly = np.array(page_poly, dtype=np.float32)

            # Build a Shapely polygon and compute its convex hull
            poly = Polygon(page_poly)
            if not poly.is_valid:
                poly = poly.convex_hull

            # Add buffer and simplify only when using boxes as masks
            if use_boxes_as_masks:
                poly = poly.buffer(1.0)
                poly = poly.simplify(tolerance=1.0, preserve_topology=True)

            # Extract the exterior coords (drop the closing point)
            smoothed_coords = list(poly.exterior.coords)[:-1]
            self.logger.info(f"Final polygon: {len(smoothed_coords)} points")

            # Create CoordsType from the smoothed polygon
            region_coords = CoordsType(
                points_from_polygon(smoothed_coords),
                conf=score
            )

            cat = category.split(':')
            self.logger.info(f"Category split: {category} to {cat}")

            # TextLine case
            if cat[0] == 'TextLine':
                # Lines must live inside TextRegions (or TextRegions inside tables),
                # not directly on the page or table.
                if level == 'page':
                    self.logger.warning(f"Got TextLine category {category} on page level: lines must be created inside regions. Skipping.")
                    continue

                # Decide where to attach the line
                parent_region = None

                if isinstance(segment, TextRegionType):
                    # Normal case running on a TextRegion
                    parent_region = segment

                elif isinstance(segment, TableRegionType):
                    # Table level
                    # Create or reuse a TextRegion inside the TableRegion to hold lines
                    existing_text_regions = segment.get_TextRegion() or []
                    if existing_text_regions:
                        parent_region = existing_text_regions[0]
                    else:
                        # Create one container TextRegion for all lines in this table
                        tr_id = f'{segment.id}_textregion'
                        # Use the table's Coords as the region's Coords
                        tr_coords = segment.get_Coords() or region_coords
                        parent_region = TextRegionType(id=tr_id, Coords=tr_coords)
                        segment.add_TextRegion(parent_region)
                        self.logger.info(f"Created TextRegion {tr_id} inside TableRegion {segment.id} to hold TextLines")
                else:
                    self.logger.warning(f"Skipping TextLine detection on unsupported segment type {segment.id} ({segment.__class__.__name__})")
                    continue

                if parent_region is None:
                    self.logger.warning(f"Could not determine parent TextRegion for TextLine in segment {segment.id} ({segment.__class__.__name__}): skipping")
                    continue

                line_no += 1
                line_id = f'{parent_region.id}_line_{line_no:04d}'
                line = TextLineType(id=line_id, Coords=region_coords)

                if len(cat) > 1:
                    line.set_custom(cat[1])

                parent_region.add_TextLine(line)
                self.logger.info(f"Added TextLine {line_id} to {parent_region.id}")
                continue

            cat2class = {
                'AdvertRegion': AdvertRegionType,
                'ChartRegion': ChartRegionType,
                'ChemRegion': ChemRegionType,
                'CustomRegion': CustomRegionType,
                'GraphicRegion': GraphicRegionType,
                'ImageRegion': ImageRegionType,
                'LineDrawingRegion': LineDrawingRegionType,
                'MapRegion': MapRegionType,
                'MathsRegion': MathsRegionType,
                'MusicRegion': MusicRegionType,
                'NoiseRegion': NoiseRegionType,
                'SeparatorRegion': SeparatorRegionType,
                'TableRegion': TableRegionType,
                'TextRegion': TextRegionType,
                'UnknownRegion': UnknownRegionType,
            }

            try:
                regiontype = cat2class[cat[0]]
            except KeyError:
                raise ValueError(f"Invalid region type {cat[0]}")

            region_no += 1
            region_id = f'region{region_no:04d}_{cat[0]}'
            region = regiontype(id=region_id, Coords=region_coords)

            # Set subtype
            if len(cat) > 1:
                try:
                    subtype_map = {
                        TextRegionType: TextTypeSimpleType,
                        GraphicRegionType: GraphicsTypeSimpleType,
                        ChartRegionType: ChartTypeSimpleType
                    }
                    if regiontype in subtype_map:
                        # subtype_map[regiontype](cat[1])
                        region.set_type(cat[1])
                    else:
                        region.set_custom(cat[1])
                except (KeyError, ValueError):
                    region.set_custom(cat[1])

            self.logger.info(f"About to add {cat[0]} to {segment.__class__.__name__}")

            # Check if the segment has the required add method
            add_method = f'add_{cat[0]}'
            if not hasattr(segment, add_method):
                self.logger.error(f"Segment {segment.__class__.__name__} does not have method {add_method}!")
                continue

            getattr(segment, add_method)(region)

            self.logger.info(f"=== Completed iteration {idx + 1}/{len(masks)} ===")
            self.logger.info(f"Detected {category} {region_no} ({score}) on {segtype} {segment.id}")

        # Debug visualization if requested
        if self.parameter.get('debug_img') != 'none':
            vis_img = array_raw.copy()
            for mask, class_id in zip(masks, classes):
                color = np.random.randint(0, 255, 3).tolist()
                mask_indices = mask.astype(np.uint8)
                vis_img[mask_indices > 0] = vis_img[mask_indices > 0] * 0.5 + np.array(color) * 0.5

            altimg = AlternativeImageType(comments='debug')
            segment.add_AlternativeImage(altimg)
            return OcrdPageResultImage(
                Image.fromarray(vis_img.astype(np.uint8)),
                ('' if isinstance(segment, PageType) else '_' + segment.id) + '.IMG-DEBUG',
                altimg)

        return None
