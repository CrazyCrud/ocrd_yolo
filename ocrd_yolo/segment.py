from ocrd_utils import getLogger, coordinates_of_segment, coordinates_for_segment, points_from_polygon
from ocrd_models.ocrd_page import TableRegionType, TextRegionType, CoordsType, PageType, AlternativeImageType
from ocrd import Processor, OcrdPageResult, OcrdPageResultImage
from ultralytics import YOLO
import cv2, numpy as np
from PIL import Image
import torch
from shapely.geometry import Polygon
from shapely.ops import unary_union
from .nms import postprocess_nms, postprocess_morph
from .utils import polygon_for_parent, make_valid, _ensure_consistent_crops


class Yolo2Segment(Processor):
    pass
