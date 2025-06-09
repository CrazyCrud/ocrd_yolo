import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from ocrd_utils import (
    resource_filename,
    getLogger,
    pushd_popd,
    coordinates_of_segment,
    coordinates_for_segment,
    crop_image,
    points_from_polygon,
    polygon_from_points,
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


def polygon_for_parent(polygon, parent):
    """Clip polygon to parent polygon range.

    (Should be moved to ocrd_utils.coordinates_for_segment.)
    """
    childp = Polygon(polygon)
    if isinstance(parent, PageType):
        if parent.get_Border():
            parentp = Polygon(polygon_from_points(parent.get_Border().get_Coords().points))
        else:
            parentp = Polygon([[0, 0], [0, parent.get_imageHeight()],
                               [parent.get_imageWidth(), parent.get_imageHeight()],
                               [parent.get_imageWidth(), 0]])
    else:
        parentp = Polygon(polygon_from_points(parent.get_Coords().points))
    # ensure input coords have valid paths (without self-intersection)
    # (this can happen when shapes valid in floating point are rounded)
    childp = make_valid(childp)
    parentp = make_valid(parentp)
    if not childp.is_valid:
        return None
    if not parentp.is_valid:
        return None
    # check if clipping is necessary
    if childp.within(parentp):
        return childp.exterior.coords[:-1]
    # clip to parent
    interp = childp.intersection(parentp)
    # post-process
    if interp.is_empty or interp.area == 0.0:
        return None
    if interp.type == 'GeometryCollection':
        # heterogeneous result: filter zero-area shapes (LineString, Point)
        interp = unary_union([geom for geom in interp.geoms if geom.area > 0])
    if interp.type == 'MultiPolygon':
        # homogeneous result: construct convex hull to connect
        # FIXME: construct concave hull / alpha shape
        interp = interp.convex_hull
    if interp.minimum_clearance < 1.0:
        # follow-up calculations will necessarily be integer;
        # so anticipate rounding here and then ensure validity
        interp = Polygon(np.round(interp.exterior.coords))
        interp = make_valid(interp)
    return interp.exterior.coords[:-1]  # keep open


def make_valid(polygon):
    for split in range(1, len(polygon.exterior.coords) - 1):
        if polygon.is_valid or polygon.simplify(polygon.area).is_valid:
            break
        # simplification may not be possible (at all) due to ordering
        # in that case, try another starting point
        polygon = Polygon(polygon.exterior.coords[-split:] + polygon.exterior.coords[:-split])
    for tolerance in range(1, int(polygon.area)):
        if polygon.is_valid:
            break
        # simplification may require a larger tolerance
        polygon = polygon.simplify(tolerance)
    return polygon


def _ensure_consistent_crops(image_raw, image_bin):
    # workaround for OCR-D/core#687:
    if 0 < abs(image_raw.width - image_bin.width) <= 2:
        diff = image_raw.width - image_bin.width
        if diff > 0:
            image_raw = crop_image(
                image_raw,
                (int(np.floor(diff / 2)), 0,
                 image_raw.width - int(np.ceil(diff / 2)),
                 image_raw.height))
        else:
            image_bin = crop_image(
                image_bin,
                (int(np.floor(-diff / 2)), 0,
                 image_bin.width - int(np.ceil(-diff / 2)),
                 image_bin.height))
    if 0 < abs(image_raw.height - image_bin.height) <= 2:
        diff = image_raw.height - image_bin.height
        if diff > 0:
            image_raw = crop_image(
                image_raw,
                (0, int(np.floor(diff / 2)),
                 image_raw.width,
                 image_raw.height - int(np.ceil(diff / 2))))
        else:
            image_bin = crop_image(
                image_bin,
                (0, int(np.floor(-diff / 2)),
                 image_bin.width,
                 image_bin.height - int(np.ceil(-diff / 2))))
    return image_raw, image_bin
