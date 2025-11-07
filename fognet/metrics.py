"""
Some metrics are adapted from Cellpose (https://github.com/MouseLand/cellpose). 
Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer , Michael Rariden and Marius Pachitariu.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix 
import concurrent.futures
from scipy.spatial.distance import cdist
from numba import jit
from tqdm import tqdm
import torch

def mask_ious(masks_true, masks_pred):
    """Return best-matched masks."""
    iou = _intersection_over_union(masks_true, masks_pred)[1:, 1:]
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= 0.5).astype(float) - iou / (2 * n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    iout = np.zeros(masks_true.max())
    iout[true_ind] = iou[true_ind, pred_ind]
    preds = np.zeros(masks_true.max(), "int")
    preds[true_ind] = pred_ind + 1
    return iout, preds


# def boundary_scores(masks_true, masks_pred, scales):
#     """
#     Calculate boundary precision, recall, and F-score.

#     Args:
#         masks_true (list): List of true masks.
#         masks_pred (list): List of predicted masks.
#         scales (list): List of scales.

#     Returns:
#         tuple: A tuple containing precision, recall, and F-score arrays.
#     """
#     diams = [utils.diameters(lbl)[0] for lbl in masks_true]
#     precision = np.zeros((len(scales), len(masks_true)))
#     recall = np.zeros((len(scales), len(masks_true)))
#     fscore = np.zeros((len(scales), len(masks_true)))
#     for j, scale in enumerate(scales):
#         for n in range(len(masks_true)):
#             diam = max(1, scale * diams[n])
#             rs, ys, xs = utils.circleMask([int(np.ceil(diam)), int(np.ceil(diam))])
#             filt = (rs <= diam).astype(np.float32)
#             otrue = utils.masks_to_outlines(masks_true[n])
#             otrue = convolve(otrue, filt)
#             opred = utils.masks_to_outlines(masks_pred[n])
#             opred = convolve(opred, filt)
#             tp = np.logical_and(otrue == 1, opred == 1).sum()
#             fp = np.logical_and(otrue == 0, opred == 1).sum()
#             fn = np.logical_and(otrue == 1, opred == 0).sum()
#             precision[j, n] = tp / (tp + fp)
#             recall[j, n] = tp / (tp + fn)
#         fscore[j] = 2 * precision[j] * recall[j] / (precision[j] + recall[j])
#     return precision, recall, fscore



def compute_binary_iou(predictions, ground_truth,device='cuda',disable_tqdm=False):
    """
    Compute Intersection over Union (IoU) for binary masks.
    
    Args:
        predictions List[(torch.Tensor)]: Predicted binary masks.
        ground_truth List[(torch.Tensor)]: Ground truth binary masks.
        device (str): Device to perform the computation on ('cpu' or 'cuda').

        
    Returns:
        torch.Tensor: IoU scores for each mask.
    """

    iou_scores = []
    for pred, gt in tqdm(zip(predictions, ground_truth), desc="Computing IoU",disable=disable_tqdm):
        # Ensure both tensors are on the same device
    
        pred = torch.tensor(pred).to(device)
        gt = torch.tensor(gt).to(device)

        # Ensure both tensors are binary masks
        pred = (pred > 0.5).float()
        gt = (gt > 0.5).float()

        # Compute intersection and union
        intersection = torch.sum(pred * gt)
        union = torch.sum(pred) + torch.sum(gt) - intersection

        # Compute IoU
        iou = intersection / (union + 1e-6)  # Add a small value to avoid division by zero
        iou_scores.append(iou.item())

    # Average IoU across all masks
    iou_scores = torch.tensor(iou_scores, device=device)
    if len(iou_scores) == 0:
        return torch.tensor(0.0, device=device)  # Return 0 if no masks are provided
    iou_scores = torch.mean(iou_scores)
    return iou_scores


def _label_overlap(masks_true, masks_pred):
    return csr_matrix((np.ones((masks_true.size,), "int"),
                       (masks_true.flatten(), masks_pred.flatten())),
                      shape=(masks_true.max() + 1, masks_pred.max() + 1))


def aggregated_jaccard_index(masks_true, masks_pred):
    """
    AJI = intersection of all matched masks / union of all masks

    Args:
        masks_true (list of np.ndarrays (int) or np.ndarray (int)):
            where 0=NO masks; 1,2... are mask labels
        masks_pred (list of np.ndarrays (int) or np.ndarray (int)):
            np.ndarray (int) where 0=NO masks; 1,2... are mask labels

    Returns:
        aji (float): aggregated jaccard index for each set of masks
    """
    aji = np.zeros(len(masks_true))
    for n in range(len(masks_true)):
        iout, preds = mask_ious(masks_true[n], masks_pred[n])
        inds = np.arange(0, masks_true[n].max(), 1, int)
        overlap = _label_overlap(masks_true[n], masks_pred[n])
        union = np.logical_or(masks_true[n] > 0, masks_pred[n] > 0).sum()
        overlap = overlap[inds[preds > 0] + 1, preds[preds > 0].astype(int)]
        aji[n] = overlap.sum() / union
    return aji


def average_precision(masks_true, masks_pred, threshold=[0.5, 0.75, 0.9]):
    """
    Average precision estimation: AP = TP / (TP + FP + FN)

    This function is based heavily on the *fast* stardist matching functions
    (https://github.com/mpicbg-csbd/stardist/blob/master/stardist/matching.py)

    Args:
        masks_true (list of np.ndarrays (int) or np.ndarray (int)):
            where 0=NO masks; 1,2... are mask labels
        masks_pred (list of np.ndarrays (int) or np.ndarray (int)):
            np.ndarray (int) where 0=NO masks; 1,2... are mask labels

    Returns:
        ap (array [len(masks_true) x len(threshold)]):
            average precision at thresholds
        tp (array [len(masks_true) x len(threshold)]):
            number of true positives at thresholds
        fp (array [len(masks_true) x len(threshold)]):
            number of false positives at thresholds
        fn (array [len(masks_true) x len(threshold)]):
            number of false negatives at thresholds
    """
    not_list = False
    if not isinstance(masks_true, list):
        masks_true = [masks_true]
        masks_pred = [masks_pred]
        not_list = True
    if not isinstance(threshold, list) and not isinstance(threshold, np.ndarray):
        threshold = [threshold]

    if len(masks_true) != len(masks_pred):
        raise ValueError(
            "metrics.average_precision requires len(masks_true)==len(masks_pred)")

    ap = np.zeros((len(masks_true), len(threshold)), np.float32)
    tp = np.zeros((len(masks_true), len(threshold)), np.float32)
    fp = np.zeros((len(masks_true), len(threshold)), np.float32)
    fn = np.zeros((len(masks_true), len(threshold)), np.float32)
    f1 = np.zeros((len(masks_true), len(threshold)), np.float32)
    n_true = np.array([len(np.unique(mt)) - 1 for mt in masks_true])
    n_pred = np.array([len(np.unique(mp)) - 1 for mp in masks_pred])

    for n in range(len(masks_true)):
        #_,mt = np.reshape(np.unique(masks_true[n], return_index=True), masks_pred[n].shape)
        if n_pred[n] > 0:
            iou = _intersection_over_union(masks_true[n], masks_pred[n])[1:, 1:]
            for k, th in enumerate(threshold):
                tp[n, k] = _true_positive(iou, th)
        fp[n] = n_pred[n] - tp[n]
        fn[n] = n_true[n] - tp[n]
        f1[n] = 2 * tp[n] / (2 * tp[n] + fp[n] + fn[n])
        ap[n] = tp[n] / (tp[n] + fp[n] + fn[n])

    if not_list:
        ap, tp, fp, fn ,f1= ap[0], tp[0], fp[0], fn[0],f1[0]
    return ap, tp, fp, fn, f1




def _intersection_over_union(masks_true, masks_pred):
    """Calculate the intersection over union of all mask pairs.

    Parameters:
        masks_true (np.ndarray, int): Ground truth masks, where 0=NO masks; 1,2... are mask labels.
        masks_pred (np.ndarray, int): Predicted masks, where 0=NO masks; 1,2... are mask labels.

    Returns:
        iou (np.ndarray, float): Matrix of IOU pairs of size [x.max()+1, y.max()+1].

    How it works:
        The overlap matrix is a lookup table of the area of intersection
        between each set of labels (true and predicted). The true labels
        are taken to be along axis 0, and the predicted labels are taken
        to be along axis 1. The sum of the overlaps along axis 0 is thus
        an array giving the total overlap of the true labels with each of
        the predicted labels, and likewise the sum over axis 1 is the
        total overlap of the predicted labels with each of the true labels.
        Because the label 0 (background) is included, this sum is guaranteed
        to reconstruct the total area of each label. Adding this row and
        column vectors gives a 2D array with the areas of every label pair
        added together. This is equivalent to the union of the label areas
        except for the duplicated overlap area, so the overlap matrix is
        subtracted to find the union matrix.
    """
    if masks_true.size != masks_pred.size:
        raise ValueError("masks_true.size != masks_pred.size")
    overlap = _label_overlap(masks_true, masks_pred)
    overlap = overlap.toarray()
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou


def _true_positive(iou, th):
    """Calculate the true positive at threshold th.

    Args:
        iou (float, np.ndarray): Array of IOU pairs.
        th (float): Threshold on IOU for positive label.

    Returns:
        tp (float): Number of true positives at threshold.

    How it works:
        (1) Find minimum number of masks.
        (2) Define cost matrix; for a given threshold, each element is negative
            the higher the IoU is (perfect IoU is 1, worst is 0). The second term
            gets more negative with higher IoU, but less negative with greater
            n_min (but that's a constant...).
        (3) Solve the linear sum assignment problem. The costs array defines the cost
            of matching a true label with a predicted label, so the problem is to 
            find the set of pairings that minimizes this cost. The scipy.optimize
            function gives the ordered lists of corresponding true and predicted labels. 
        (4) Extract the IoUs from these pairings and then threshold to get a boolean array
            whose sum is the number of true positives that is returned. 
    """
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= th).astype(float) - iou / (2 * n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    match_ok = iou[true_ind, pred_ind] >= th
    tp = match_ok.sum()
    return tp





###  DISTANCE BASED METRICS ###

def relabel_sequential(label_field, offset=1):
    """Relabel arbitrary labels to {`offset`, ... `offset` + number_of_labels}.

    This function also returns the forward map (mapping the original labels to
    the reduced labels) and the inverse map (mapping the reduced labels back
    to the original ones).

    Parameters
    ----------
    label_field : numpy array of int, arbitrary shape
        An array of labels, which must be non-negative integers.
    offset : int, optional
        The return labels will start at `offset`, which should be
        strictly positive.

    Returns
    -------
    relabeled : numpy array of int, same shape as `label_field`
        The input label field with labels mapped to
        {offset, ..., number_of_labels + offset - 1}.
        The data type will be the same as `label_field`, except when
        offset + number_of_labels causes overflow of the current data type.
    forward_map : numpy array of int, shape ``(label_field.max() + 1,)``
        The map from the original label space to the returned label
        space. Can be used to re-apply the same mapping. See examples
        for usage. The data type will be the same as `relabeled`.
    inverse_map : 1D numpy array of int, of length offset + number of labels
        The map from the new label space to the original space. This
        can be used to reconstruct the original label field from the
        relabeled one. The data type will be the same as `relabeled`.

    Notes
    -----
    The label 0 is assumed to denote the background and is never remapped.

    The forward map can be extremely big for some inputs, since its
    length is given by the maximum of the label field. However, in most
    situations, ``label_field.max()`` is much smaller than
    ``label_field.size``, and in these cases the forward map is
    guaranteed to be smaller than either the input or output images.

    """
    offset = int(offset)
    if offset <= 0:
        raise ValueError("Offset must be strictly positive.")
    if np.min(label_field) < 0:
        raise ValueError("Cannot relabel array that contains negative values.")
    max_label = int(label_field.max()) # Ensure max_label is an integer
    if not np.issubdtype(label_field.dtype, np.integer):
        new_type = np.min_scalar_type(max_label)
        label_field = label_field.astype(new_type)
    labels = np.unique(label_field)
    labels0 = labels[labels != 0]
    new_max_label = offset - 1 + len(labels0)
    new_labels0 = np.arange(offset, new_max_label + 1)
    output_type = label_field.dtype
    required_type = np.min_scalar_type(new_max_label)
    if np.dtype(required_type).itemsize > np.dtype(label_field.dtype).itemsize:
        output_type = required_type
    forward_map = np.zeros(max_label + 1, dtype=output_type)
    forward_map[labels0] = new_labels0
    inverse_map = np.zeros(new_max_label + 1, dtype=output_type)
    inverse_map[offset:] = labels0
    relabeled = forward_map[label_field]
    return relabeled, forward_map, inverse_map


def normalize(X, q_min, q_max):
    mini = np.quantile(X, q_min)
    maxi = np.quantile(X, q_max)
    X = np.clip(X, mini, maxi)

    X -= mini
    X /= (maxi - mini)

    return X.astype(np.float32)

# fast implem with jit
@jit(nopython=True)
def get_positions(labels):
    n = labels.max() + 1
    positions = np.zeros((n, 2), dtype=np.float32)
    m_00 = np.zeros(n, dtype=np.uint)
    m_01 = np.zeros(n, dtype=np.uint)
    m_10 = np.zeros(n, dtype=np.uint)

    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            m_00[labels[i, j]] += 1
            m_01[labels[i, j]] += i
            m_10[labels[i, j]] += j

    positions[:, 0] = m_01 / m_00
    positions[:, 1] = m_10 / m_00
    return positions




def compute_dist(labels: np.ndarray, pred: np.ndarray) -> np.ndarray:
    assert labels.shape == pred.shape


    positions_true = get_positions(labels)
    positions_pred = get_positions(pred)

    return cdist(positions_true, positions_pred)


def _compute_dist(Y_true: np.ndarray, Y_pred: np.ndarray, thresh=2.0) -> np.ndarray:
    Y_true = relabel_sequential(Y_true)[0]
    Y_pred = relabel_sequential(Y_pred)[0]
    dist = compute_dist(Y_true, Y_pred)[1:, 1:]
    dist[dist>thresh] = 100
    n_true, n_pred = dist.shape
    i, j = linear_sum_assignment(dist)
    return dist,n_true, n_pred,i,j

def matching_dataset_dist(Y_trues, Y_preds, thresh = 2.,use_futures=True):
    stats = {"tp": 0, "n_true": 0, "n_pred": 0}
    if use_futures:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures=[executor.submit(_compute_dist, Y_true, Y_pred,thresh) for Y_true, Y_pred in zip(Y_trues, Y_preds)]
        for future in concurrent.futures.as_completed(futures):
            dist, n_true, n_pred, i, j = future.result()

            stats["n_true"] += int(n_true)
            stats["n_pred"] += int(n_pred)
            stats["tp"] += int((dist[i, j] < thresh).sum())
    else:
        for Y_true, Y_pred in zip(Y_trues, Y_preds):
            dist, n_true, n_pred, i, j = _compute_dist(Y_true, Y_pred,thresh)

            stats["n_true"] += int(n_true)
            stats["n_pred"] += int(n_pred)
            stats["tp"] += int((dist[i, j] < thresh).sum())
    
    if stats["n_pred"] == 0:
        stats["n_pred"] = np.inf
    if stats["n_true"] == 0:
        stats["n_true"] = np.inf
        
    stats["fp"] = stats["n_pred"] - stats["tp"]
    stats["fn"] = stats["n_true"] - stats["tp"]
    stats["precision"] = stats["tp"] / stats["n_pred"]
    stats["recall"] = stats["tp"] / stats["n_true"]
    stats["f1"] = 2*stats["tp"] / (stats["n_pred"] + stats["n_true"])
    
    return stats
