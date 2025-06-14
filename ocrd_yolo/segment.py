from __future__ import absolute_import

import cv2
import numpy as np
from PIL import Image
import torch
from typing import Optional

from ultralytics import YOLO
from shapely.geometry import Polygon

from ocrd_utils import (
    getLogger,
    coordinates_of_segment,
    coordinates_for_segment,
    points_from_polygon,
    polygon_from_points
)
from ocrd_models.ocrd_page import (
    OcrdPage,
    PageType,
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
    max_workers = 1  # GPU context sharable across not forks

    @property
    def executable(self):
        return 'ocrd-yolo-segment'

    def setup(self):
        # Device selection
        if self.parameter['device'] == 'cpu' or not torch.cuda.is_available():
            device = "cpu"
        else:
            device = self.parameter['device']
        self.logger.info("Using device %s", device)

        # Load model
        model_weights = self.resolve_resource(self.parameter['model_weights'])
        self.logger.info("Loading YOLOv11 weights from %s", model_weights)
        self.model = YOLO(model_weights)
        self.model.to(device)

        # Parameters
        self.min_confidence = float(self.parameter.get('min_confidence', 0.5))
        self.categories = self.parameter['categories']

        # Validate categories match model classes
        model_classes = self.model.model.names if hasattr(self.model.model, 'names') else {}
        self.logger.info(f"Model has {len(model_classes)} classes")

    def process_page_pcgts(self, *input_pcgts: Optional[OcrdPage], page_id: Optional[str] = None) -> OcrdPageResult:
        """Use YOLO to segment each page into regions."""
        pcgts = input_pcgts[0]
        result = OcrdPageResult(pcgts)
        level = self.parameter['operation_level']

        page = pcgts.get_Page()
        page_image_raw, page_coords, page_image_info = self.workspace.image_from_page(
            page, page_id, feature_filter='binarized')

        # For morphological post-processing, we need the binarized image
        if self.parameter['postprocessing'] != 'none':
            page_image_bin, _, _ = self.workspace.image_from_page(
                page, page_id, feature_selector='binarized')
            page_image_raw, page_image_bin = _ensure_consistent_crops(
                page_image_raw, page_image_bin)
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

        if zoom < 2.0:
            zoomed = zoom / 2.0
            self.logger.info("scaling %dx%d image by %.2f", page_image_raw.width, page_image_raw.height, zoomed)
        else:
            zoomed = 1.0

        for segment in ([page] if level == 'page' else
        page.get_AllRegions(depth=1, classes=['Table'])):
            # Get existing regions
            def at_segment(region):
                return region.parent_object_ is segment

            regions = list(filter(at_segment, page.get_AllRegions()))

            if isinstance(segment, PageType):
                image_raw = page_image_raw
                image_bin = page_image_bin
                coords = page_coords
            else:
                image_raw, coords = self.workspace.image_from_segment(
                    segment, page_image_raw, page_coords, feature_filter='binarized')
                if self.parameter['postprocessing'] != 'none':
                    image_bin, _ = self.workspace.image_from_segment(
                        segment, page_image_bin, page_coords)
                    image_raw, image_bin = _ensure_consistent_crops(
                        image_raw, image_bin)
                else:
                    image_bin = image_raw

            # Ensure RGB
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

            image = self._process_segment(segment, regions, coords, array_raw, array_bin, zoomed, page_id)
            if image:
                result.images.append(image)

        return result

    def _process_segment(self, segment, ignore, coords, array_raw, array_bin, zoomed, page_id) -> Optional[
        OcrdPageResultImage]:
        self.logger = getLogger('processor.Yolo2Segment')
        segtype = segment.__class__.__name__[:-4]
        segment.set_custom('coords=%s' % coords['transform'])
        height, width = array_raw.shape[:2]
        postprocessing = self.parameter['postprocessing']

        # Estimate scale for morphological operations
        scale = 43
        if postprocessing in ['full', 'only-morph']:
            _, components = cv2.connectedComponents(array_bin.astype(np.uint8))
            _, counts = np.unique(components, return_counts=True)
            if counts.shape[0] > 1:
                counts = np.sqrt(3 * counts)
                counts = counts[(5 < counts) & (counts < 100)]
                scale = int(np.median(counts))
                self.logger.debug("estimated scale: %d", scale)

        # Run YOLO inference
        results = self.model(array_raw, conf=self.min_confidence, verbose=False)

        if not results or not results[0].boxes:
            self.logger.warning("Detected no regions on %s '%s'", segtype, segment.id)
            return None

        # Extract detections from YOLO results
        result = results[0]
        boxes = result.boxes

        # Get masks, scores, and classes
        # YOLOv11 always provides segmentation masks
        masks = result.masks.data.cpu().numpy()
        # Resize masks to original image size if needed
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
            keep_indices = [i for i, cls in enumerate(classes) if cls < len(self.categories) and self.categories[cls]]
            if not keep_indices:
                self.logger.warning("No detections for selected categories on %s '%s'", segtype, segment.id)
                return None
            masks = masks[keep_indices]
            scores = scores[keep_indices]
            classes = classes[keep_indices]

        # Handle existing regions for NMS
        if len(ignore):
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

        # Apply post-processing
        if postprocessing in ['full', 'only-nms']:
            scores, classes, masks = postprocess_nms(
                scores, classes, masks, array_bin, self.categories,
                min_confidence=self.min_confidence, nproc=8, logger=self.logger)

        if postprocessing in ['full', 'only-morph']:
            _, components = cv2.connectedComponents(array_bin.astype(np.uint8))
            scores, classes, masks = postprocess_morph(
                scores, classes, masks, components, nproc=8, logger=self.logger)

        # Remove placeholder for existing regions
        if len(ignore):
            scores = scores[1:]
            classes = classes[1:]
            masks = masks[1:]

        # Convert masks to regions
        region_no = 0
        for mask, class_id, score in zip(masks, classes, scores):
            category = self.categories[class_id]

            # Find contours
            invalid = True
            for _ in range(10):
                contours, _ = cv2.findContours(mask.astype(np.uint8),
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) == 1 and len(contours[0]) > 3:
                    invalid = False
                    break
                mask = cv2.dilate(mask.astype(np.uint8),
                                  np.ones((scale, scale), np.uint8)) > 0

            if invalid:
                self.logger.warning("Ignoring non-contiguous (%d) region for %s", len(contours), category)
                continue

            region_polygon = contours[0][:, 0, :]  # x,y order
            if zoomed != 1.0:
                region_polygon = region_polygon / zoomed

            # Transform coordinates
            region_polygon = coordinates_for_segment(region_polygon, None, coords)
            region_polygon = polygon_for_parent(region_polygon, segment)
            if region_polygon is None:
                self.logger.warning("Ignoring extant region for %s", category)
                continue

            # Create region
            region_coords = CoordsType(points_from_polygon(region_polygon), conf=score)
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

            cat = category.split(':')
            try:
                regiontype = cat2class[cat[0]]
            except KeyError:
                raise ValueError(f"Invalid region type {cat[0]}")

            region_no += 1
            region_id = f'region{region_no:04d}_{cat[0]}'
            region = regiontype(id=region_id, Coords=region_coords)

            # Set subtype if specified
            if len(cat) > 1:
                try:
                    subtype_map = {
                        TextRegionType: TextTypeSimpleType,
                        GraphicRegionType: GraphicsTypeSimpleType,
                        ChartRegionType: ChartTypeSimpleType
                    }
                    if regiontype in subtype_map:
                        subtype_map[regiontype](cat[1])
                        region.set_type(cat[1])
                    else:
                        region.set_custom(cat[1])
                except (KeyError, ValueError):
                    region.set_custom(cat[1])

            getattr(segment, f'add_{cat[0]}')(region)
            self.logger.info("Detected %s region%04d (p=%.2f) on %s '%s'",
                             category, region_no, score, segtype, segment.id)

        # Debug visualization if requested
        if self.parameter.get('debug_img') != 'none':
            # Create visualization
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