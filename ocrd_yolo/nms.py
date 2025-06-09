import multiprocessing as mp
from shapely.geometry import Polygon
from shapely.ops import unary_union
import numpy as np
import cv2
import ctypes

from ocrd_utils import (
    getLogger
)

# when doing Numpy postprocessing, enlarge masks via
# outer (convex) instead of inner (concave) hull of
# corresponding connected components
NP_POSTPROCESSING_OUTER = False
# when pruning overlapping detections (in either mode),
# require at least this share of the area to be redundant
RECALL_THRESHOLD = 0.8
# when finalizing contours of detections (in either mode),
# snap to connected components overlapping by this share
# (of component area), i.e. include if larger and exclude
# if smaller than this much
IOCC_THRESHOLD = 0.4
# when finalizing contours of detections (in either mode),
# add this many pixels in each direction
FINAL_DILATION = 4


def overlapmasks_init(masks_array, masks_shape):
    global shared_masks
    global shared_masks_shape
    shared_masks = masks_array
    shared_masks_shape = masks_shape


def overlapmasks(i, j):
    # is i redundant w.r.t. j (i.e. j already covers most of its area)
    masks = np.ctypeslib.as_array(shared_masks).reshape(shared_masks_shape)
    imask = masks[i]
    jmask = masks[j]
    intersection = np.count_nonzero(imask * jmask)
    if not intersection:
        return False
    base = np.count_nonzero(imask)
    if intersection / base > RECALL_THRESHOLD:
        return True
    return False


def morphmasks_init(masks_array, masks_shape, components_array, components_shape):
    global shared_masks
    global shared_masks_shape
    global shared_components
    global shared_components_shape
    shared_masks = masks_array
    shared_masks_shape = masks_shape
    shared_components = components_array
    shared_components_shape = components_shape


def morphmasks(instance):
    masks = np.ctypeslib.as_array(shared_masks).reshape(shared_masks_shape)
    components = np.ctypeslib.as_array(shared_components).reshape(shared_components_shape)
    mask = masks[instance]
    # find closure in connected components
    complabels = np.unique(mask * components)
    left, top, w, h = cv2.boundingRect(mask.astype(np.uint8))
    right = left + w
    bottom = top + h
    if NP_POSTPROCESSING_OUTER:
        # overwrite pixel mask from (padded) outer bbox
        for label in complabels:
            if not label:
                continue  # bg/white
            leftc, topc, wc, hc = cv2.boundingRect((components == label).astype(np.uint8))
            rightc = leftc + wc
            bottomc = topc + hc
            if wc > 2 * w or hc > 2 * h:
                continue  # huge (non-text?) component
            # intersection over component too small?
            if (min(right, rightc) - max(left, leftc)) * \
                    (min(bottom, bottomc) - max(top, topc)) < IOCC_THRESHOLD * wc * hc:
                continue  # too little overlap
            newleft = min(left, leftc)
            newtop = min(top, topc)
            newright = max(right, rightc)
            newbottom = max(bottom, bottomc)
            if (newright - newleft) > 2 * w or (newbottom - newtop) > 1.5 * h:
                continue  #
            left = newleft
            top = newtop
            right = newright
            bottom = newbottom
            w = right - left
            h = bottom - top
        left = max(0, left - FINAL_DILATION)
        top = max(0, top - FINAL_DILATION)
        right = min(mask.shape[1], right + FINAL_DILATION)
        bottom = min(mask.shape[0], bottom + FINAL_DILATION)
        mask[top:bottom, left:right] = True

    else:
        # fill pixel mask from (padded) inner bboxes
        for label in complabels:
            if not label:
                continue  # bg/white
            suppress = False
            leftc, topc, wc, hc = cv2.boundingRect((components == label).astype(np.uint8))
            rightc = leftc + wc
            bottomc = topc + hc
            if wc > 2 * w or hc > 2 * h:
                # huge (non-text?) component
                suppress = True
            if (min(right, rightc) - max(left, leftc)) * \
                    (min(bottom, bottomc) - max(top, topc)) < IOCC_THRESHOLD * wc * hc:
                # intersection over component too small
                suppress = True
            newleft = min(left, leftc)
            newtop = min(top, topc)
            newright = max(right, rightc)
            newbottom = max(bottom, bottomc)
            if (newright - newleft) > 2 * w or (newbottom - newtop) > 1.5 * h:
                # huge (non-text?) component
                suppress = True
            elif (newright - newleft) < 1.1 * w and (newbottom - newtop) < 1.1 * h:
                suppress = False
            if suppress:
                leftc = min(mask.shape[1], leftc + FINAL_DILATION)
                topc = min(mask.shape[0], topc + FINAL_DILATION)
                rightc = max(0, rightc - FINAL_DILATION)
                bottomc = max(0, bottomc - FINAL_DILATION)
                mask[topc:bottomc, leftc:rightc] = False
            else:
                leftc = max(0, leftc - FINAL_DILATION)
                topc = max(0, topc - FINAL_DILATION)
                rightc = min(mask.shape[1], rightc + FINAL_DILATION)
                bottomc = min(mask.shape[0], bottomc + FINAL_DILATION)
                mask[topc:bottomc, leftc:rightc] = True
                left = newleft
                top = newtop
                right = newright
                bottom = newbottom
                w = right - left
                h = bottom - top


def tonumpyarray_with_shape(mp_arr, shape):
    return np.frombuffer(mp_arr, dtype=np.dtype(mp_arr)).reshape(shape)


def postprocess_nms(scores, classes, masks, page_array_bin, categories, min_confidence=0.5, nproc=8, logger=None):
    """Apply geometrical post-processing to raw detections: remove overlapping candidates via non-maximum suppression across classes.

    Implement via Numpy routines.
    """
    if logger is None:
        logger = getLogger('ocrd.processor.Yolo2Segment')
    # apply IoU-based NMS across classes
    assert masks.dtype == bool
    instances = np.arange(len(masks))
    instances_i, instances_j = np.meshgrid(instances, instances, indexing='ij')
    combinations = list(zip(*np.where(instances_i != instances_j)))
    shared_masks = mp.sharedctypes.RawArray(ctypes.c_bool, masks.size)
    shared_masks_np = tonumpyarray_with_shape(shared_masks, masks.shape)
    np.copyto(shared_masks_np, masks * page_array_bin)
    with mp.Pool(processes=nproc,  # to be refined via param
                 initializer=overlapmasks_init,
                 initargs=(shared_masks, masks.shape)) as pool:
        # multiprocessing for different combinations of array slices (pure)
        overlapping_combinations = pool.starmap(overlapmasks, combinations)
    overlaps = np.zeros((len(masks), len(masks)), bool)
    for (i, j), overlapping in zip(combinations, overlapping_combinations):
        if overlapping:
            overlaps[i, j] = True
    # find best-scoring instance per class
    bad = np.zeros_like(instances, bool)
    for i in np.argsort(-scores):
        score = scores[i]
        mask = masks[i]
        assert mask.shape[:2] == page_array_bin.shape[:2]
        ys, xs = mask.nonzero()
        assert xs.any() and ys.any(), "instance has empty mask"
        bbox = [xs.min(), ys.min(), xs.max(), ys.max()]
        class_id = classes[i]
        if class_id < 0:
            logger.debug("ignoring existing region at %s", str(bbox))
            continue
        category = categories[class_id]
        if scores[i] < min_confidence:
            logger.debug("Ignoring instance for %s with too low score %.2f", category, score)
            bad[i] = True
            continue
        count = np.count_nonzero(mask)
        if count < 10:
            logger.warning("Ignoring too small (%dpx) region for %s", count, category)
            bad[i] = True
            continue
        worse = score < scores
        if np.any(worse & overlaps[i]):
            logger.debug("Ignoring instance for %s with %.2f overlapping better neighbour",
                         category, score)
            bad[i] = True
        else:
            logger.debug("post-processing prediction for %s at %s area %d score %f",
                         category, str(bbox), count, score)
    # post-process detections morphologically and decode to region polygons
    # does not compile (no OpenCV support):
    keep = np.nonzero(~ bad)[0]
    if not keep.size:
        return [], [], []
    keep = sorted(keep, key=lambda i: scores[i], reverse=True)
    scores = scores[keep]
    classes = classes[keep]
    masks = masks[keep]
    return scores, classes, masks


def postprocess_morph(scores, classes, masks, components, nproc=8, logger=None):
    """Apply morphological post-processing to raw detections: extend masks to avoid chopping off fg connected components.

    Implement via Numpy routines.
    """
    if logger is None:
        logger = getLogger('ocrd.processor.Detectron2Segment')
    shared_masks = mp.sharedctypes.RawArray(ctypes.c_bool, masks.size)
    shared_components = mp.sharedctypes.RawArray(ctypes.c_int32, components.size)
    shared_masks_np = tonumpyarray_with_shape(shared_masks, masks.shape)
    shared_components_np = tonumpyarray_with_shape(shared_components, components.shape)
    np.copyto(shared_components_np, components, casting='equiv')
    np.copyto(shared_masks_np, masks)
    with mp.Pool(processes=nproc,  # to be refined via param
                 initializer=morphmasks_init,
                 initargs=(shared_masks, masks.shape,
                           shared_components, components.shape)) as pool:
        # multiprocessing for different slices of array (in-place)
        pool.map(morphmasks, range(masks.shape[0]))
    masks = tonumpyarray_with_shape(shared_masks, masks.shape)
    return scores, classes, masks
