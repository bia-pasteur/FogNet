import numpy as np
import cv2
import torch
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from tqdm import tqdm
from scipy.ndimage import label,binary_dilation
import os
import tifffile

def watershed_instanciation(seg_masks, peak_local_max_footprint=np.ones((5, 5)),markers_structure=np.ones((3, 3)),disable_tqdm=False):
    """
    Apply watershed algorithm to binary segmentation masks.
    Args:
        seg_masks List[torch.Tensor or np.array ]: List of binary segmentation masks.
    Returns:
        np.ndarray: Labels for each instance in the segmentation mask.
    """
    labels = []
    for seg_mask in tqdm(seg_masks, desc="Applying Watershed", disable=disable_tqdm):
        if not isinstance(seg_mask, np.ndarray):
            seg_mask = seg_mask.cpu().numpy().squeeze()
        elif len(seg_mask.shape) > 2:
            raise ValueError("Input mask should be 2D or 3D, not multi-channel.")
       
        seg_mask = seg_mask > 0  # Ensure binary mask
        distance = ndi.distance_transform_edt(seg_mask)
        coords = peak_local_max(distance, footprint=peak_local_max_footprint, labels=seg_mask)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        mask = ndi.binary_dilation(mask, iterations=1)
        markers, _ = ndi.label(mask, structure=markers_structure)
        label = watershed(-distance, markers, mask=seg_mask)
        labels.append(label)
    return labels


def watershed_instanciation_from_pseudo_probs(out,min_distance=2,threshold_abs=0.5):
    labels_PPWS = []
    print(out.shape)
    for pseudo_prob in tqdm(out, desc="Applying Watershed from Pseudo Probabilities"):

        pseudo_prob_np=pseudo_prob.numpy()
        binary_mask=(pseudo_prob_np > 0.5).astype(np.uint8)

        coordinates = peak_local_max(pseudo_prob_np, min_distance=min_distance, threshold_abs=threshold_abs, labels=binary_mask)
        mask=np.zeros_like(pseudo_prob_np,dtype=bool)
        mask[tuple(coordinates.T)] = True
        mask = binary_dilation(mask, iterations=1)
        markers, _ = label(mask, structure=np.ones((3,3)))
        label_PPWS = watershed(-pseudo_prob_np, markers, mask=binary_mask)
        labels_PPWS.append(label_PPWS)
    return labels_PPWS


def create_sparse_dataset(data_folder,window_size=3):
    masks_folder=os.path.join(data_folder, "tracks")
    frame_ids = []
    dataset=[]
    for mask_file in sorted(os.listdir(masks_folder)):
        if not mask_file.endswith(".tiff"):
            continue
        frame_id = int(mask_file.split(".")[0])
        frame_ids.append(frame_id)
        print(f"Processing frame {frame_id}...")
        mask= tifffile.imread(os.path.join(masks_folder, mask_file))

        frames_list = []
        for ids in range(frame_id - window_size//2, frame_id + window_size//2 + 1):
            frame= tifffile.imread(os.path.join(data_folder, f"{ids:03d}.tiff"))
            frames_list.append(frame)

        dataset.append((frames_list, mask))

    return dataset
 

def train_val_test_split(dataset, val_ratio=0.0, test_ratio=0.2, seed=42):
    np.random.seed(seed)
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    test_size = int(len(dataset) * test_ratio)
    val_size = int(len(dataset) * val_ratio)
    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size + val_size]
    train_indices = indices[test_size + val_size:]
    return (
        [dataset[i] for i in train_indices],
        [dataset[i] for i in val_indices],
        [dataset[i] for i in test_indices],
    )




def warp_flow(flow, img1=None, img2=None, interpolation=cv2.INTER_LINEAR):
    """Use remap to warp flow, generating a new image. cv2.remap(frame1, map_x_backward, map_y_backward, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

Args:
    flow (np.ndarray): flow
    img1 (np.ndarray, optional): previous frame
    img2 (np.ndarray, optional): next frame
Returns:
    warped image
If img1 is input, the output will be img2_warped, but there will be multiple pixels corresponding to a single pixel, resulting in sparse holes. 
If img2 is input, the output will be img1_warped, and there will be no sparse holes. The latter approach is preferred.
    """
    h, w, _ = flow.shape
    remap_flow = flow.transpose(2, 0, 1)
    remap_xy = np.float32(np.mgrid[:h, :w][::-1])
    if img1 is not None:
        uv_new = (remap_xy + remap_flow).round().astype(np.int32)
        mask = (uv_new[0] >= 0) & (uv_new[1] >= 0) & (uv_new[0] < w) & (uv_new[1] < h)
        uv_new_ = uv_new[:, mask]
        remap_xy[:, uv_new_[1], uv_new_[0]] = remap_xy[:, mask]
        remap_x, remap_y = remap_xy
        img2_warped = cv2.remap(img1, remap_x, remap_y, interpolation)
        mask_remaped = np.zeros((h, w), np.bool_)
        mask_remaped[uv_new_[1], uv_new_[0]] = True
        img2_warped[~mask_remaped] = 0
        return img2_warped
    elif img2 is not None:
        remap_x, remap_y = np.float32(remap_xy + remap_flow)
        return cv2.remap(img2, remap_x, remap_y, interpolation)
    

def average_prob_over_window_of(prob_maps, frame_n, flows, nim=3, sigma=1):
    """
    Average the probs over a larger temporal window
    """
    frame=prob_maps[frame_n]
    
    window=[]
    for i in range(-(nim//2) +(1-nim%2), nim//2+1):
        if frame_n+i>=0 and frame_n+i<len(prob_maps):
            prob_map=prob_maps[frame_n+i]
            print(i)
            if i<0:
                for j in range(abs(i)):
                    flow=flows[frame_n+i+j]
                    prob_map=warp_flow(flow=flow, img1=prob_map)[...,np.newaxis]
            elif i>0:
                for j in range(abs(i)):
                    flow=flows[frame_n+i-1-j]
                    prob_map = warp_flow(flow=flow,img2=prob_map)[...,np.newaxis]

            window.append(prob_map*gaussian(i, sigma))
        else:
            window.append(frame*gaussian(i, sigma))
        


    window=np.array(window)

    window=window.mean(axis=0)

    # Normalize the window
    window=window/window.max()


    return window

def gaussian(x, sigma):
    """
    Computes the Gaussian function value for a given x and standard deviation sigma.

    Parameters:
    x : float or np.ndarray
        The input value(s) at which to evaluate the Gaussian.
    sigma : float
        The standard deviation of the Gaussian.

    Returns:
    float or np.ndarray
        The value of the Gaussian function at x.
    """
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(- (x**2) / (2 * sigma**2))

def gaussian_torch(x, sigma):
    """
    Computes the Gaussian function value for a given x and standard deviation sigma using PyTorch.

    Parameters:
    x : torch.Tensor
        The input value(s) at which to evaluate the Gaussian.
    sigma : float
        The standard deviation of the Gaussian.

    Returns:
    torch.Tensor
        The value of the Gaussian function at x.
    """
    pi = torch.tensor(torch.pi)
    x = torch.tensor(x, dtype=torch.float32)
    
    return (1 / (sigma * torch.sqrt(2 * pi))) * torch.exp(- (x**2) / (2 * sigma**2))

def sigma_to_sigma_param(sigma, n_channels=3, tolerance=1e-2):
    """
    Converts the actual standard deviation of the Gaussian to a learned sigma parameter.

    Parameters:
    sigma : float
        The standard deviation of the Gaussian.
    n_channels : int
        The number of channels in the input data (default is 3).
    tolerance : float
        The tolerance level for the Gaussian (default is 1e-2).

    Returns:
    float
        The learned parameter to compute the standard deviation of the Gaussian.
    """
    pre_sigma = sigma * (np.sqrt(-2 * np.log(1 - tolerance))) / (n_channels // 2)
    sigma_param = np.log(pre_sigma / (1 - pre_sigma))

    return sigma_param


def sigma_param_to_sigma(sigma_param,n_channels=3,tolerance=1e-2):
    """
    Converts a learned sigma parameter to the actual standard deviation of the Gaussian.

    Parameters:
    sigma_param : float
        The learned parameter to compute the standard deviation of the Gaussian.
    n_channels : int
        The number of channels in the input data (default is 3).
    tolerance : float
        The tolerance level for the Gaussian (default is 1e-2).

    Returns:
    float
        The computed standard deviation of the Gaussian.
    """
    pre_sigma = torch.sigmoid(sigma_param)
    sigma= pre_sigma*(n_channels//2)/(np.sqrt(-2*np.log(1-tolerance)))
    
    return sigma.item()

def gaussian_from_sigma_param(i, sigma_param,n_channels=3,tolerance=1e-2):

    """
    Computes the Gaussian function value for a given x and standard deviation sigma.


    Parameters:
    i : torch.tensor      The input value(s) at which to evaluate the Gaussian.
    sigma_param : float    The learned parameter to compute the standard deviation of the Gaussian.

    Returns:
    torch.tensor         The value of the Gaussian function at x.
    """
    pre_sigma = torch.sigmoid(sigma_param)
    sigma= pre_sigma*(n_channels//2)/(np.sqrt(-2*np.log(1-tolerance)))
    
    return gaussian_torch(i, sigma)

