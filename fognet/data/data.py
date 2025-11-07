from matplotlib import image
import torch
import lightning as L
from torch.utils.data import DataLoader, Dataset
import numpy as np
import tifffile
import os
import cv2
import math
import itertools
from typing import List, Union, Sequence, Optional, Generator, cast, TypeVar
import warnings
from torch.utils.data import Subset

class DefaultDataset(Dataset):
    """
    A default PyTorch Dataset for loading images and corresponding segmentation masks.
    Args:
        images (list or np.ndarray): A list or numpy array of image tensors.
        masks (list or np.ndarray): A list or numpy array of mask tensors.
        transform (callable, optional): Optional transform to be applied on the images and masks.

    Returns:
        tuple: A tuple containing the image tensor and the corresponding mask tensor.
        The image tensor has shape (C, H, W) and the mask tensor has shape (1, H, W).
    """

    def __init__(self, images, masks=None, transform=None, in_channels=1):
        self.images = images
        self.masks = masks
        self.transform = transform
        self.in_channels = in_channels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        if self.in_channels > 1:
            # If in_channels > 1, we assume images are already in (C, H, W) format
            return self._get_multichannel_item(idx)
        else:
            return self._get_single_channel_item(idx)

    def _get_single_channel_item(self, idx):

        image = self.images[idx]
        if self.masks is not None:
            mask = self.masks[idx]
            # Ensure mask is binary (0 or 1)
            mask = (mask > 0).astype(np.float32)
        else:
            print(
                "DefaultDatasetWarning: No masks provided. The dataset will return empty masks. This should only be used for testing purposes."
            )
            mask = np.zeros_like(image)

        # Ensure image and mask are tensors
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(
                0
            )  # Add channel dimension
        else:
            image = image.float().unsqueeze(0)

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        else:
            mask = mask.float().unsqueeze(0)

        # Optional transform logic
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

    def _get_multichannel_item(self, idx):
        image = self.images[idx]
        if self.masks is not None:
            mask = self.masks[idx]
            # Ensure mask is binary (0 or 1)
            mask = (mask > 0).astype(np.float32)
        else:
            print(
                "DefaultDatasetWarning: No masks provided. The dataset will return empty masks. This should only be used for testing purposes."
            )
            mask = np.zeros_like(image)

        # Ensure image and mask are tensors
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32)
        else:
            image = image.float()
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.float32)
        else:
            mask = mask.float()

        assert (
            image.ndim == 3
        ), "Image tensor must have shape (C, H, W) for multichannel input."
        assert (
            mask.ndim == 3
        ), "Mask tensor must have shape (C, H, W) for multichannel input."

        return image, mask  # Add channel dimension to mask

    def get_images(self):
        """
        Retrieve all images from the dataset as a list of tensors.
        Returns:
            list: A list of image tensors, each with shape (C, H, W).
        """

        images = []
        for i in range(len(self)):
            image = self.__getitem__(i)[0]
            images.append(image)
        return images

    def get_masks(self):
        """
        Retrieve all masks from the dataset as a list of tensors.
        Returns:
            list: A list of mask tensors, each with shape (1, H, W).
        """

        masks = []
        for i in range(len(self)):
            mask = self.__getitem__(i)[1]
            masks.append(mask)
        return masks


class DefaultFogDataset(Dataset):
    """A default PyTorch Dataset for loading images and corresponding segmentation masks for multichannel predictions.
    Expects images and masks to be single channel, the dataset will convert them to multichannel by stacking them along the channel dimension.
    Expects frames to be in the order of the video, i.e. the first frame is the first image in the list, the second frame is the second image in the list, etc.
    Performs constant padding on the left and right side of the video to ensure that the model can predict on the first and last frames.


    Args:
        images (list or np.ndarray): A list or numpy array of image arrays.
        masks (list or np.ndarray): A list or numpy array of mask arrays.
        transform (callable, optional): Optional transform to be applied on the images and masks.
        in_channels (int): Number of channels (size of window) for the experiment. Default is 3.

    Returns:
        tuple: A tuple containing the image tensor and the corresponding mask tensor, each of length of video frames.
        The image tensor has shape (C, H, W) and the mask tensor has shape (1, H, W).

    """

    def __init__(
        self, images, masks=None, transform=None, in_channels=3, precalculate_flow=False
    ):
        self.in_channels = in_channels
        self.transform = transform
        self.precalculate_flow = precalculate_flow
        self.masks = masks
        self.images = images

        if self.masks is None:
            print(
                "DefaultFogDataset: No masks provided. The dataset will return empty masks. This should only be used for testing purposes."
            )
            self.masks = [
                np.zeros_like(self.images[i]) for i in range(len(self.images))
            ]

        self.add_padding()

        if self.precalculate_flow:
            self.forward_flow, self.backward_flow = self._precalculate_flow()

    def __len__(self):
        return len(self.masks)

    def _precalculate_flow(self):
        print("FogDataset: Precalculating optical flow using Farneback method...")
        forward_flow = []
        backward_flow = []
        for i in range(len(self.images) - 1):
            prev_frame = self.images[i]
            next_frame = self.images[i + 1]
            flow_forward = cv2.calcOpticalFlowFarneback(
                prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            flow_backward = cv2.calcOpticalFlowFarneback(
                next_frame, prev_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            forward_flow.append(flow_forward)
            backward_flow.append(flow_backward)

        return forward_flow, backward_flow

    def add_padding(self):
        for _ in range(self.in_channels // 2):
            self.images = np.concatenate(
                (
                    self.images[0][np.newaxis, ...],
                    self.images,
                    self.images[-1][np.newaxis, ...],
                ),
                axis=0,
            )

    def _make_multichannel(self, data):
        """
        Convert a list or numpy array of single-channel images or masks into a multichannel format.
        Args:
            data (list or np.ndarray): A list or numpy array of single-channel images or masks.
        Returns:
            np.ndarray: A numpy array of shape (N, C, H, W) where N is the number of samples, C is the number of channels,
                        H is the height and W is the width.
        """

        if isinstance(data, list):
            data = np.array(data)
        data = data.squeeze()
        return data.reshape(self.in_channels, *data.shape[1:])

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        Args:
            idx (int): Index of the item to retrieve.
        Returns:
            tuple: A tuple containing the image tensor and the corresponding mask tensor.
                   The image tensor has shape (C, H, W) and the mask tensor has shape (1, H, W).
        """
        # Get the multichannel images and masks
        if self.in_channels == 1:
            # If in_channels is 1, we treat each image as a single channel
            image = self.images[idx]
            mask = self.masks[idx]
        else:

            image = self._make_multichannel(self.images[idx : idx + self.in_channels])
            mask = self.masks[idx]

        # Ensure image and mask are tensors
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32)
        else:
            image = image.float()

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.float32)
        else:
            mask = mask.float()

        # Optional transform logic
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        if not self.precalculate_flow:
            return image, mask

        # If flow is precalculated, return it as well
        else:
            flows = []
            for channel in range(self.in_channels - 1):
                if channel < self.in_channels // 2:
                    flows.append(self.backward_flow[idx + channel])
                else:
                    flows.append(self.forward_flow[idx + channel])

            return image, mask, flows


class SparseFogDataset(Dataset):
    """A default PyTorch Dataset for loading images and corresponding segmentation masks for multichannel predictions.
    This dataset is for sparsly sampled videos, where the frames are not consecutive. It expect the dataset to be a list of (images, mask) tuples, where images is a list of frames corresponding to the window size.
    The central frame is the annotated frame, and the surrounding frames are used for context.
    Args:
        dataset (list): A list of tuples (images, mask), where images is a list of frames and mask is the corresponding segmentation mask.
        in_channels (int): Number of channels (size of window) for the experiment. Default is 3.
        precalculate_flow (bool): Whether to precalculate optical flow between frames. Default is False.


    """

    def __init__(self, dataset, in_channels=3, precalculate_flow=False):
        self.images = []
        self.masks = []
        self.in_channels = in_channels
        self.precalculate_flow = precalculate_flow
        for images, mask in dataset:
            # normalize images to [0,1]

            self.images.append(images)
            self.masks.append(mask)
            assert (
                len(images) == in_channels
            ), f"Expected {in_channels} images, got {len(images)}"
        if self.precalculate_flow:
            self.flows = self._precalculate_flow()

    def _precalculate_flow(self):
        print("SparseFogDataset: Precalculating optical flow using Farneback method...")
        flows = []

        for i in range(len(self.images)):
            flow_per_image = []
            for j in range(self.in_channels - 1):
                prev_frame = self.images[i][j]
                next_frame = self.images[i][j + 1]
                if j < self.in_channels // 2:
                    flow = cv2.calcOpticalFlowFarneback(
                        next_frame, prev_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                else:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                flow_per_image.append(flow)
            flows.append(flow_per_image)

        return flows

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        image = (np.array(self.images[idx])).squeeze()
        mask = self.masks[idx]

        # Ensure image and mask are tensors
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32)
        else:
            image = image.float()

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.float32)
        else:
            mask = mask.float()
        if len(image.shape) == 2:
            image = image.unsqueeze(0)

        if not self.precalculate_flow:
            return image, mask

        else:
            return image, mask, self.flows[idx]


class AugmentedDataset(Dataset):
    """Wrap a dataset and apply an augmenter callable to (image, mask).
    Supports augmenters that follow albumentations API (augmenter(image=..., mask=...))
    or simple callables that take/return (image, mask) as tensors.
    Preserves extra returned items (e.g. flows) from the underlying dataset.
    """

    def __init__(self, base_ds: Dataset, augmenter):
        self.base_ds = base_ds
        if type(augmenter) is not list:
            augmenter = [augmenter]
        self.augmenter = augmenter

    def __len__(self):
        return len(self.base_ds)

    def _to_numpy(self, img):
        if isinstance(img, torch.Tensor):
            arr = img.detach().cpu().numpy().squeeze()
        else:
            arr = np.array(img).squeeze()
        # channel-first -> channel-last for augmenters like albumentations
        return arr.copy()

    def _to_tensor(self, arr):
        if isinstance(arr, torch.Tensor):
            return arr.float()
        else:
            return torch.tensor(arr, dtype=torch.float32)

    def __getitem__(self, idx):
        item = self.base_ds[idx]
        # handle (image, mask) or (image, mask, flows)
        if not isinstance(item, (list, tuple)):
            raise RuntimeError("Wrapped dataset must return tuple (image, mask[, ...])")
        image, mask = item

        # augmenter(image=..., mask=...)
        for aug in self.augmenter:
            image_t, mask_t = aug(image, mask)

        return (image_t, mask_t)


class SegmentationDataModule(L.LightningDataModule):
    """
    A PyTorch Lightning DataModule for loading training and validation datasets for image segmentation tasks.
    Args:
        train_ds (Dataset): The training dataset.
        val_ds (Dataset): The validation dataset.
        batch_size (int): Batch size for the data loaders.
    """

    def __init__(
        self,
        train_ds=None,
        val_ds=None,
        test_ds=None,
        augmenter=None,
        batch_size=16,
        train_val_test_split=False,
        shuffle=True,
        seed=None,
    ):

        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds

        self.augmenter = augmenter

        self.batch_size = batch_size
        self.train_val_test_split = train_val_test_split
        self.shuffle = shuffle
        self.seed = seed
        if self.seed is not None:
            torch.manual_seed(self.seed)

        if self.train_val_test_split:
            print(
                "SegmentationDataModuleWarning: train_val_test_split is set to True, the train_ds, val_ds and test_ds will be split from the train_ds. If you want to use custom train_ds, val_ds and test_ds, set train_val_test_split to False."
            )
            if self.train_ds is None:
                raise ValueError(
                    "train_ds must be provided when train_val_test_split is True"
                )
            if self.val_ds is not None or self.test_ds is not None:
                print(
                    "SegmentationDataModuleWarning: val_ds and test_ds will be ignored when train_val_test_split is True"
                )
            # Split the train_ds into train, val and test sets
            print(
                "Splitting the train_ds into train, val and test sets of lengths ",
                f"{int(len(self.train_ds)*0.8)}, {int(len(self.train_ds)*0.1)}, {len(self.train_ds)-int(len(self.train_ds)*0.8)-int(len(self.train_ds)*0.1)}",
            )
            self.train_ds, self.val_ds, self.test_ds = _video_dataset_split(
                self.train_ds,
                [
                    int(len(self.train_ds) * 0.8),
                    int(len(self.train_ds) * 0.1),
                    len(self.train_ds)
                    - int(len(self.train_ds) * 0.8)
                    - int(len(self.train_ds) * 0.1),
                ],
            )

        if self.augmenter is not None:
            if self.train_ds is not None:
                print(
                    "Applying augmenter to train_dataset. Make sure optical flow is not precalculated."
                )
                self.train_ds = AugmentedDataset(self.train_ds, self.augmenter)

    def train_dataloader(self):

        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=self.shuffle
        )

    def val_dataloader(self):

        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self):

        return DataLoader(self.test_ds, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size)


class FogDataModule(SegmentationDataModule):
    """
    A PyTorch Lightning DataModule for loading training and validation datasets for image segmentation tasks.
    This is a specialized version of SegmentationDataModule for Flow Guided Temporal Averaging experiments.
    Args:
        root_folder (str): The root folder containing the dataset. Expected folder structure is:
        window_size (int): Size of the window for the FOG experiment. Default is 3.
        subfolders=["train", "val", "test"]. The train, val and test folders should contain a video.tiff file and a folder with the masks. Can be renamed to any other name or None, to ignore one or more of the splits.
        train_val_test_split (bool): If True, the train_ds, val_ds and test_ds will be split from the train_ds. If False, the train_ds, val_ds and test_ds will be used as provided.
        batch_size (int): Batch size for the data loaders.
        augmenter (callable, optional): Optional data augmentation function to be applied on the images and masks.
        seed (int, optional): Random seed for reproducibility. If None, no seed is set.

    Returns:
        A PyTorch Lightning DataModule with train, val and test datasets.
        The datasets are instances of DefaultFogDataset, which is a subclass of PyTorch Dataset.
        The images and masks are loaded from the video.tiff file and the folder,
        and the images are expected to be in the format (C, H, W) where C is the number of channels, H is the height and W is the width.
        The masks are expected to be in the format (1, H, W) where 1 is the number of channels (binary masks).
        The datasets can be used with PyTorch DataLoader for training and validation and testing.
    """

    def __init__(
        self,
        root_folder,
        window_size=3,
        subfolders=["train", "val", "test"],
        mask_dir_name="tracks",
        train_val_test_split=False,
        max_length_of_training_samples=200,
        max_length_of_validation_samples=50,
        max_length_of_testing_samples=200,
        batch_size=1,
        augmenter=None,
        shuffle=True,
        seed=None,
        precalculate_flow=False,
    ):
        if not os.path.exists(root_folder):
            raise ValueError(f"Root folder {root_folder} does not exist.")

        datasets = []
        train_val_or_test = 0  # 1=train, 2=val, 3=test
        for subfolder in subfolders:
            train_val_or_test += 1
            if subfolder is None:
                datasets.append(None)
                continue
            if not os.path.exists(os.path.join(root_folder, subfolder)):
                print(
                    f"FTGADataModuleWarning: {subfolder} folder does not exist in {root_folder}. Skipping {subfolder} split. Make sure this is intended."
                )
                datasets.append(None)
            else:
                video_path = os.path.join(root_folder, subfolder, "video.tiff")
                mask_path = os.path.join(root_folder, subfolder, mask_dir_name)
                if not os.path.exists(video_path):
                    raise ValueError(
                        f"Video file {video_path} does not exist in {subfolder} folder."
                    )
                if not os.path.exists(mask_path):
                    print(
                        f"FTGADataModuleWarning: Mask folder {mask_path} does not exist in {subfolder} folder. Using empty masks for {subfolder} split. Make sure this is intended."
                    )
                    masks = None
                else:
                    masks = _masks_folder_to_masks(mask_path)
                video = tifffile.imread(video_path)

                if train_val_or_test == 1:  # train
                    if (
                        max_length_of_training_samples is not None
                        and len(video) > max_length_of_training_samples
                    ):
                        print(
                            f"FogDataModuleWarning: Training video in {subfolder} is longer than max_length_of_training_samples ({len(video)}>{max_length_of_training_samples}). Truncating the dataset for faster training. Set max_length_of_training_samples to None to use the full dataset."
                        )
                        video = video[:max_length_of_training_samples]
                        if masks is not None:
                            masks = masks[:max_length_of_training_samples]
                elif train_val_or_test == 2:  # val
                    if (
                        max_length_of_validation_samples is not None
                        and len(video) > max_length_of_validation_samples
                    ):
                        print(
                            f"FogDataModuleWarning: Validation video in {subfolder} is longer than max_length_of_validation_samples ({len(video)}>{max_length_of_validation_samples}). Truncating the dataset for faster validation. Set max_length_of_validation_samples to None to use the full dataset."
                        )
                        video = video[:max_length_of_validation_samples]
                        if masks is not None:
                            masks = masks[:max_length_of_validation_samples]
                elif train_val_or_test == 3:  # test
                    if (
                        max_length_of_testing_samples is not None
                        and len(video) > max_length_of_testing_samples
                    ):
                        print(
                            f"FogDataModuleWarning: Testing video in {subfolder} is longer than max_length_of_testing_samples ({len(video)}>{max_length_of_testing_samples}). Truncating the dataset for faster testing. Set max_length_of_testing_samples to None to use the full dataset."
                        )
                        video = video[:max_length_of_testing_samples]
                        if masks is not None:
                            masks = masks[:max_length_of_testing_samples]

                if window_size == 1:
                    datasets.append(DefaultDataset(video, masks))
                else:
                    datasets.append(
                        DefaultFogDataset(
                            video,
                            masks,
                            in_channels=window_size,
                            precalculate_flow=precalculate_flow,
                        )
                    )

        print("######## Successfully loaded datasets ########")

        print(f"Root folder: {root_folder}")
        print(
            f"Loaded data with shape: {datasets[0][0][0].shape if datasets[0] is not None else 'N/A'} and masks with shape: {datasets[0][0][1].shape if datasets[0] is not None else 'N/A'}"
        )
        print(
            f"Train dataset length: {len(datasets[0]) if datasets[0] is not None else 'N/A'}"
        )
        print(
            f"Val dataset length: {len(datasets[1]) if datasets[1] is not None else 'N/A'}"
        )
        print(
            f"Test dataset length: {len(datasets[2]) if datasets[2] is not None else 'N/A'}"
        )
        print("###############################################")
        super().__init__(
            *datasets,
            augmenter=augmenter,
            batch_size=batch_size,
            train_val_test_split=train_val_test_split,
            shuffle=shuffle,
            seed=seed,
        )


def _masks_folder_to_masks(mask_dir):
    """Load tif image masks from a folder and return them as a list of numpy arrays.
    folder should contain numbered tif images.
    """

    masks = []
    for mask_file in sorted(os.listdir(mask_dir)):
        if mask_file.endswith(".tif") or mask_file.endswith(".tiff"):
            mask_path = os.path.join(mask_dir, mask_file)
            mask = tifffile.imread(mask_path)
            masks.append(mask)
    masks = np.array(masks)
    return masks


_T = TypeVar("_T")
def _video_dataset_split(
    dataset: Dataset[_T],
    lengths: Sequence[Union[int, float]]) -> List[Subset[_T]]:
    r"""
    Splits a dataset into non-overlapping new datasets of given lengths. Does not randomize the indices before splitting, to keep the video order intact.
    If the sum of the input lengths is less than the length of the dataset,
    the last split will be smaller than the specified length.

    Modified from PyTorch's `torch.utils.data.random_split`
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(
                    f"Length of split at index {i} is 0. "
                    f"This might result in an empty dataset."
                )

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):  # type: ignore[arg-type]
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )
    indices = list(range(len(dataset)))  # type: ignore[arg-type]
    lengths = cast(Sequence[int], lengths)
    return [
        Subset(dataset, indices[offset - length : offset])
        for offset, length in zip(itertools.accumulate(lengths), lengths)
    ]
