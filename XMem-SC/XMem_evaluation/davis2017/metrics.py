import math
import numpy as np
import cv2
import skimage.measure as measure
import torch.nn.functional as F
import torch

def db_eval_blob(annotations, segmentations, void_pixels=None):
    """ Compute instance similarity as the Blob Index.
    Arguments:
        annotation   (ndarray): binary annotation map.
        segmentation (ndarray): binary segmentation map.
        void_pixels  (ndarray): optional mask with void pixels
    Return:
        blob (float): region similarity
    """
    assert annotations.shape == segmentations.shape, \
        f'Annotation({annotations.shape}) and segmentation:{segmentations.shape} dimensions do not match.'
    annotations = annotations.astype(np.bool)
    segmentations = segmentations.astype(np.bool)
    blob_ious = []
    for annotation, segmentation in zip(annotations, segmentations):
        blob_annotation = measure.label(annotation, connectivity = 2)
        blob_segmentation = measure.label(segmentation, connectivity = 2)
        
        one_hot_anno = F.one_hot(torch.tensor(blob_annotation).to(torch.int64), num_classes=np.max(blob_annotation)+1).permute(2, 0, 1)
        one_hot_anno_wo_bg = np.array(one_hot_anno[1:])
        one_hot_segm = F.one_hot(torch.tensor(blob_segmentation).to(torch.int64), num_classes=np.max(blob_segmentation)+1).permute(2, 0, 1)
        one_hot_segm_wo_bg = np.array(one_hot_segm[1:])
        
        instance_iou = []
        if one_hot_anno_wo_bg.shape[0] == 0:
            blob_ious.append(0)
            continue
        for i in range(one_hot_anno_wo_bg.shape[0]):
            gt_instance = one_hot_anno_wo_bg[i]
            max_iou = 0
            max_iou_idx = -1
            if one_hot_segm_wo_bg.shape[0] == 0:
                break
            for j in range(one_hot_segm_wo_bg.shape[0]):
                pred_instance = one_hot_segm_wo_bg[j]
                inter = np.sum((pred_instance & gt_instance), axis=(-2, -1))
                union = np.sum((pred_instance | gt_instance), axis=(-2, -1))
                iou = (inter / union)
                if iou > max_iou:
                    max_iou = iou
                    max_iou_idx = j
            instance_iou.append(max_iou)
            one_hot_segm_wo_bg = np.concatenate([one_hot_segm_wo_bg[:max_iou_idx],one_hot_segm_wo_bg[(max_iou_idx+1):]])
        if len(instance_iou) != 0:
            blob_ious.append(np.mean(instance_iou))
        else:
            blob_ious.append(0)
    return blob_ious

def db_eval_iou(annotation, segmentation, void_pixels=None):
    """ Compute region similarity as the Jaccard Index.
    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.
        void_pixels  (ndarray): optional mask with void pixels

    Return:
        jaccard (float): region similarity
    """
    assert annotation.shape == segmentation.shape, \
        f'Annotation({annotation.shape}) and segmentation:{segmentation.shape} dimensions do not match.'
    annotation = annotation.astype(np.bool)
    segmentation = segmentation.astype(np.bool)

    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape, \
            f'Annotation({annotation.shape}) and void pixels:{void_pixels.shape} dimensions do not match.'
        void_pixels = void_pixels.astype(np.bool)
    else:
        void_pixels = np.zeros_like(segmentation)

    
    inters = np.sum((segmentation & annotation) & np.logical_not(void_pixels), axis=(-2, -1))
    union = np.sum((segmentation | annotation) & np.logical_not(void_pixels), axis=(-2, -1))

    j = inters / union
    if j.ndim == 0:
        j = 1 if np.isclose(union, 0) else j
    else:
        j[np.isclose(union, 0)] = 1
    return j


def db_eval_boundary(annotation, segmentation, void_pixels=None, bound_th=0.008):
    assert annotation.shape == segmentation.shape
    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape
    if annotation.ndim == 3:
        n_frames = annotation.shape[0]
        f_res = np.zeros(n_frames)
        for frame_id in range(n_frames):
            void_pixels_frame = None if void_pixels is None else void_pixels[frame_id, :, :, ]
            f_res[frame_id] = f_measure(segmentation[frame_id, :, :, ], annotation[frame_id, :, :], void_pixels_frame, bound_th=bound_th)
    elif annotation.ndim == 2:
        f_res = f_measure(segmentation, annotation, void_pixels, bound_th=bound_th)
    else:
        raise ValueError(f'db_eval_boundary does not support tensors with {annotation.ndim} dimensions')
    return f_res


def f_measure(foreground_mask, gt_mask, void_pixels=None, bound_th=0.008):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.

    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.
        void_pixels     (ndarray): optional mask with void pixels

    Returns:
        F (float): boundaries F-measure
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1
    if void_pixels is not None:
        void_pixels = void_pixels.astype(np.bool)
    else:
        void_pixels = np.zeros_like(foreground_mask).astype(np.bool)

    bound_pix = bound_th if bound_th >= 1 else \
        np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

    
    fg_boundary = _seg2bmap(foreground_mask * np.logical_not(void_pixels))
    gt_boundary = _seg2bmap(gt_mask * np.logical_not(void_pixels))

    from skimage.morphology import disk

    
    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))
    
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))

    
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)

    return F


def _seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (ndarray):	Binary boundary map.
     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
    """

    seg = seg.astype(np.bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (
        width > w | height > h | abs(ar1 - ar2) > 0.01
    ), "Can" "t convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap


if __name__ == '__main__':
    from davis2017.davis import DAVIS
    from davis2017.results import Results

    dataset = DAVIS(root='input_dir/ref', subset='val', sequences='aerobatics')
    results = Results(root_dir='examples/osvos')
    
    for seq in dataset.get_sequences():
        all_gt_masks, _, all_masks_id = dataset.get_all_masks(seq, True)
        all_gt_masks, all_masks_id = all_gt_masks[:, 1:-1, :, :], all_masks_id[1:-1]
        all_res_masks = results.read_masks(seq, all_masks_id)
        f_metrics_res = np.zeros(all_gt_masks.shape[:2])
        for ii in range(all_gt_masks.shape[0]):
            f_metrics_res[ii, :] = db_eval_boundary(all_gt_masks[ii, ...], all_res_masks[ii, ...])

    
    
