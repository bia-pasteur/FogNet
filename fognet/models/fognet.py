import torch
import torch.nn as nn
import pickle
from tqdm import tqdm
from fognet.utils import (
    watershed_instanciation,
    watershed_instanciation_from_pseudo_probs,
)
from fognet.metrics import (
    matching_dataset_dist,
    aggregated_jaccard_index,
    average_precision,
)
import scipy
from fognet.utils import gaussian_from_sigma_param, sigma_param_to_sigma
import numpy as np
import cv2
from fognet.models.unet import UNetBase, UNet
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from dataclasses import dataclass



@dataclass
class OptimizeThresholdParams:
    min_thresh: float = 0.2
    max_thresh: float = 0.8
    step: float = 0.1
    metric: str = "iou"
    val_set_size: int = 5
    disable_tqdm: bool = False
@dataclass
class PredictSegmentationParams:
    thresh: Optional[float] = None
    segmenter: str = "watershed"
    disable_tqdm: bool = False
    simple_min_area: int = 500


class FlowModule(nn.Module):
    """
    Flow Module for Flow-Guided Temporal Averaging.
    It warps the input channels according to the provided optical flow.
    Args:
        window_size (int): Number of input channels
    """

    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size

    def forward(self, x, optical_flow=None):
        if optical_flow is None:
            print("FlowModule Warning: No optical flow provided.")
            return x

        channels = self.window_size
        warped_x = []
        map_x_torch, map_y_torch = torch.meshgrid(
            torch.arange(x.shape[2]), torch.arange(x.shape[3]), indexing="ij"
        )
        for i in range(-(channels // 2) + (1 - channels % 2), channels // 2 + 1):
            current_channel = x[:, channels // 2 + i, :, :].unsqueeze(
                1
            )  # Get the current channel prediction
            if i < 0:
                for j in range(
                    abs(i)
                ):  # Forward warping : image i is warped to image i+1 using the flow from i+1 to i
                    if not isinstance(
                        optical_flow[channels // 2 + i + j], torch.Tensor
                    ):  # If the flow is not a tensor, convert it to one
                        flow = torch.tensor(
                            optical_flow[channels // 2 + i + j],
                            device=x.device,
                            dtype=torch.float32,
                        ).unsqueeze(
                            0
                        )  # Add batch dimension equal to 1 for now
                    else:  # Precalculated flow should already be a tensor
                        flow = optical_flow[channels // 2 + i + j].to(x.device)
                    backward_map_x = map_x_torch.to(x.device) + flow[..., 1]
                    backward_map_y = map_y_torch.to(x.device) + flow[..., 0]
                    backward_map_xy = torch.stack(
                        (
                            backward_map_y / current_channel.shape[3] * 2 - 1,
                            backward_map_x / current_channel.shape[2] * 2 - 1,
                        ),
                        dim=-1,
                    )
                    current_channel = torch.nn.functional.grid_sample(
                        current_channel,
                        backward_map_xy,
                        mode="bilinear",
                        align_corners=True,
                    )
            elif i > 0:
                for j in range(
                    abs(i)
                ):  # Backward warping : image i+1 is warped to image i using the flow from i to i+1
                    if not isinstance(
                        optical_flow[channels // 2 + i - 1 - j], torch.Tensor
                    ):  # If the flow is not a tensor, convert it to one
                        flow = torch.tensor(
                            optical_flow[channels // 2 + i - 1 - j],
                            device=x.device,
                            dtype=torch.float32,
                        ).unsqueeze(
                            0
                        )  # Add batch dimension equal to 1 for now
                    else:  # Precalculated flow should already be a tensor
                        flow = optical_flow[channels // 2 + i - 1 - j].to(x.device)

                    forward_map_x = map_x_torch.to(x.device) + flow[..., 1]
                    forward_map_y = map_y_torch.to(x.device) + flow[..., 0]
                    forward_map_xy = torch.stack(
                        (
                            forward_map_y / current_channel.shape[3] * 2 - 1,
                            forward_map_x / current_channel.shape[2] * 2 - 1,
                        ),
                        dim=-1,
                    )
                    current_channel = torch.nn.functional.grid_sample(
                        current_channel,
                        forward_map_xy,
                        mode="bilinear",
                        align_corners=True,
                    )

            warped_x.append(current_channel)
        out = torch.stack(warped_x, dim=1)
        return out


class PredictionHead(nn.Module):
    """
    Prediction head for Flow-Guided Temporal Averaging.
    Args:
        window_size (int): Number of input channels
        sigma_param (float or torch.Tensor, optional): Initial value for the sigma parameter. If None, it will be initialized to a value that gives an initial sigma of window_size//2.
        tolerance (float): Tolerance for max sigma value
    """

    def __init__(
        self, window_size, sigma_param=None, tolerance=1e-2
    ):  # Changed log_sigma to sigma_param
        super().__init__()
        self.window_size = window_size
        self.tolerance = tolerance

        sigma_param = (
            sigma_param
            if sigma_param is not None
            else torch.logit(
                torch.sqrt(
                    torch.tensor(-2) * torch.log(torch.tensor(1 - self.tolerance))
                )
            )
        )
        # Initialize sigma_param to a value that gives an initial sigma of channels//2
        if not isinstance(sigma_param, torch.Tensor):
            sigma_param = torch.tensor(sigma_param, dtype=torch.float32)
        self.sigma_param = nn.Parameter(sigma_param, requires_grad=True)
        print(
            f"Initialized PredictionHead with window_size={self.window_size}, initial sigma_param={self.sigma_param.item()}"
        )

    def forward(self, x):
        """
        Forward pass of the prediction head.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, window_size, height, width).
            optical_flow (torch.Tensor, optional): Optical flow tensor of shape (batch_size, 2, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1, height, width).
        """

        channels = self.window_size
        x_final = []
        weights = []
        for i in range(-channels // 2, channels // 2 + 1):
            current_channel = x[:, i + channels // 2, :, :]
            weight = gaussian_from_sigma_param(
                i, self.sigma_param, n_channels=channels, tolerance=self.tolerance
            )
            x_final.append(current_channel * weight)
            weights.append(weight)

        out = torch.stack(x_final, dim=1).sum(dim=1, keepdim=False)
        weights = torch.tensor(weights, device=x.device, dtype=torch.float32).unsqueeze(
            0
        )

        out = out / (weights.sum(dim=1, keepdim=True) + 1e-6)
        # out = torch.stack(warped_x, dim=1).sum(dim=1, keepdim=False)

        # weights = torch.tensor(weights, device=x.device, dtype=torch.float32).unsqueeze(
        #     0
        # )
        # out = out / (weights.sum(dim=1, keepdim=True) + 1e-6)

        return out


class FogNetBase(nn.Module):
    """
    FlOw-Guided Network (FogNet)

    A U-Net model with Flow-Guided Temporal Averaging (FGTA) for image segmentation tasks.
    It consists of an encoder, a bottleneck, and a decoder with skip connections, and a prediction head for FGTA.

    Args:
        window_size (int): Number of input channels
        out_channels (int): Number of output channels
        features (list): List of integers representing the number of features in each encoder block.
    """

    def __init__(self, 
                 window_size=1,
                 sigma_param=None, 
                 features=[64, 128, 256, 512]):
        super().__init__()
        self.window_size = window_size
        self.sigma_param = sigma_param
        self.unet = UNetBase(
            in_channels=window_size, out_channels=window_size, features=features
        )
        self.flow_module = FlowModule(window_size=window_size)
        self.prediction_head = PredictionHead(
            window_size=window_size, sigma_param=sigma_param
        )
        print(f"Initialized FogNet with window_size={window_size}")

    def forward(self, x, flows=None):

        if self.window_size == 1:
            return self.unet(x)

        if flows is None:
            flows = []
            channels = self.window_size

            # Calculate optical flow online for now
            # copy the imhages to the CPU and detach them from the computation graph
            images_cpu = x.clone().detach().cpu().numpy()
            # .SQUEEZE() is used to remove the batch and channel dimensions from the predictions, so that the optical flow can be calculated.
            # IF BATCH SIZE IS NOT 1, THIS WILL NOT WORK
            for channel in tqdm(
                range(channels - 1), "Calculating optical flow", disable=True
            ):
                if channel < channels // 2 + (1 - channels % 2):
                    flow = cv2.calcOpticalFlowFarneback(
                        images_cpu[:, channel + 1, :, :].squeeze(),
                        images_cpu[:, channel, :, :].squeeze(),
                        None,
                        pyr_scale=0.5,
                        levels=3,
                        winsize=15,
                        iterations=3,
                        poly_n=5,
                        poly_sigma=1.2,
                        flags=0,
                    )
                else:
                    flow = cv2.calcOpticalFlowFarneback(
                        images_cpu[:, channel, :, :].squeeze(),
                        images_cpu[:, channel + 1, :, :].squeeze(),
                        None,
                        pyr_scale=0.5,
                        levels=3,
                        winsize=15,
                        iterations=3,
                        poly_n=5,
                        poly_sigma=1.2,
                        flags=0,
                    )
                flows.append(flow)

        x = self.unet(x)
        x = self.flow_module(x, optical_flow=flows)
        return self.prediction_head(x)


class FogNet(UNet):
    """
    FlOw-Guided Network (FogNet) Lightning Module
    A U-Net model with Flow-Guided Temporal Averaging (FGTA) for image segmentation tasks.
    It consists of an encoder, a bottleneck, and a decoder with skip connections, and a prediction head for FGTA.
    Args:
        window_size (int): Number of input channels
        lr (float): Learning rate for the optimizer.
        sigma_param (float or torch.Tensor, optional): Initial value for the sigma parameter in the prediction head. If None, it will be initialized to a value that gives an initial sigma of window_size//2.
        lr_sigma (float): Learning rate for the sigma parameter optimizer.
        best_threshold (float, optional): Best threshold for binary segmentation. If None, it will be optimized after training if a validation set is provided.
        precalculate_flow (bool): If True, the datamodule must provide precalculated optical flow in the training/validation/test steps.
        max_epochs (int): Maximum number of training epochs.

    """
  
    def __init__(
        self,
        window_size: int,
        lr: float = 1e-3,
        sigma_param: Optional[Union[float, torch.Tensor]] = None,
        lr_sigma: float = 1.0,
        best_threshold: Optional[float] = None,
        precalculate_flow: bool = False,
        max_epochs: int = 30,
        optimize_threshold_params: OptimizeThresholdParams = OptimizeThresholdParams()
    ):

        self.save_hyperparameters()
        super().__init__(in_channels=window_size, out_channels=window_size, lr=lr)
        self.model = FogNetBase(window_size=window_size, sigma_param=sigma_param)
        self.sigma_param = sigma_param
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.window_size = window_size
        self.best_threshold = best_threshold
        self.lr_sigma = lr_sigma
        self.precalculate_flow = precalculate_flow
        self.max_epochs = max_epochs
        self.automatic_optimization = False
        self.optimize_threshold_params = optimize_threshold_params

        if self.precalculate_flow:
            print(
                f"FogNet Warning: precalculate_flow is set to {self.precalculate_flow}. If True, the datamodule must provide precalculated optical flow in the training/validation/test steps. This is very RAM-hungry. Make sure your configuration allows it!"
            )

    def _compute_loss(self, logits, y):
        y = (y > 0).to(torch.float32).unsqueeze(1)  # Ensure y has shape (B, 1, H, W)
        return self.loss_fn(logits, y)

    def forward(self, x, flows=None):
        if self.precalculate_flow:
            return self.model(x, flows=flows)
        else:
            return self.model(x, flows=None)

    def training_step(self, batch, batch_idx):
        torch.autograd.set_detect_anomaly(True)
        opt_main, opt_sigma = self.optimizers()

        if self.precalculate_flow:
            x, y, flows = batch
            logits = self(x, flows=flows)
        else:
            x, y = batch
            # Forward pass
            logits = self(x)

        opt_main.zero_grad()
        opt_sigma.zero_grad()

        loss = self._compute_loss(logits, y)
        self.manual_backward(loss)

        self.clip_gradients(
            opt_main, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
        )
        opt_main.step()

        self.clip_gradients(
            opt_sigma, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
        )
        opt_sigma.step()

        # Logging

        lr_main = opt_main.param_groups[0]["lr"]
        self.log("lrs/lr_main", lr_main, on_step=True, on_epoch=True, prog_bar=True)
        lr_sigma = opt_sigma.param_groups[0]["lr"]
        self.log("lrs/lr_sigma", lr_sigma, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train/sigma_param",
            self.model.prediction_head.sigma_param,
            prog_bar=True,
        )
        self.log(
            "train/sigma",
            sigma_param_to_sigma(
                self.model.prediction_head.sigma_param,
                n_channels=self.window_size,
                tolerance=1e-2,
            ),
            prog_bar=True,
        )

        self.log("train/loss", loss, on_step=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        opt_scheduler, sigma_scheduler = self.lr_schedulers()
        opt_scheduler.step()
        sigma_scheduler.step()
        return super().on_train_epoch_end()

    def validation_step(self, batch, batch_idx):
        if self.precalculate_flow:
            x, y, flows = batch
            logits = self(x, flows=flows)
        else:
            x, y = batch
            logits = self(x)
        loss = self._compute_loss(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        if batch_idx == 0 and self.logger is not None:
            preds = torch.sigmoid(logits)
            preds = (preds > 0.5).float()
            iou = aggregated_jaccard_index(
                preds.squeeze().detach().cpu().numpy().astype(np.int32),
                y.squeeze().detach().cpu().numpy().astype(np.int32),
            )
            self.log("val_iou", np.mean(iou), prog_bar=True)

        return loss

    def configure_optimizers(self):

        main_warmup_epochs = int(self.max_epochs/6)
        sigma_off_epochs = int(self.max_epochs/3)
        sigma_full_epochs = int(self.max_epochs/2)
        step_size = int(self.max_epochs/3)
        main_optimizer = torch.optim.Adam(self.model.unet.parameters(), lr=self.lr)

        # Warmup scheduler: linearly increase LR for the first 5 epochs

        def warmup_lambda(epoch):
            return min(1.0, (epoch + 1) / main_warmup_epochs)

        main_warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            main_optimizer, lr_lambda=warmup_lambda
        )

        def sigma_lambda(epoch):
            if epoch < sigma_off_epochs:
                return 0.01
            elif epoch < sigma_full_epochs:
                return 0.1
            else:
                return 1.0

        # Main scheduler: StepLR after warmup
        main_step_scheduler = torch.optim.lr_scheduler.StepLR(
            main_optimizer, step_size=step_size, gamma=0.1
        )
        # scheduler = torch.optim.lr_scheduler.ChainedScheduler([warmup_scheduler, main_scheduler])

        main_scheduler = torch.optim.lr_scheduler.SequentialLR(
            main_optimizer,
            schedulers=[main_warmup_scheduler, main_step_scheduler],
            milestones=[main_warmup_epochs],
        )

        sigma_optimizer = torch.optim.Adam(
            [self.model.prediction_head.sigma_param], lr=self.lr_sigma
        )

        sigma_scheduler = torch.optim.lr_scheduler.LambdaLR(
            sigma_optimizer, lr_lambda=sigma_lambda
        )

        return [
            {
                "optimizer": main_optimizer,
                "lr_scheduler": {"scheduler": main_scheduler},
                "name": "main_optimizer",
            },
            {
                "optimizer": sigma_optimizer,
                "lr_scheduler": {"scheduler": sigma_scheduler},
                "name": "sigma_optimizer",
            },
        ]

    def optimize_threshold(
        self,
        datamodule,
        min_thresh: float = 0.2,
        max_thresh: float = 0.8,
        step: float = 0.1,
        metric: str = "iou",
        val_set_size: int = 5,
        disable_tqdm: bool = False,
    ):
        """
        Optimize the threshold for binary segmentation using a validation set.

        Args:
            datamodule (SegmentationDataModule): The data module containing validation data.
            min_thresh (float): Minimum threshold to test.
            max_thresh (float): Maximum threshold to test.
            step (float): Step size for threshold increments.
            metric (str): Metric to optimize ('iou' only for now).
            val_set_size (int): Number of validation samples to use for threshold optimization.Ex: Setting to 5 will use the first 5 batches of the validation set.
        Returns:
            float: The optimal threshold value.
        """

        if metric != "iou" and metric != "f1":
            raise ValueError(
                "Currently, only 'iou' or 'f1' metric is supported for threshold optimization."
            )

        best_thresh = min_thresh
        best_score = 0.0

        thresholds = np.arange(min_thresh, max_thresh + step, step)
        prob_preds = []
        masks = []
        for batch_idx, batch in enumerate(datamodule.val_dataloader()):
            if batch_idx >= val_set_size:
                break

            preds = self.predict_from_dataloader(
                [batch],
                thresh=0.5,  # unused
                disable_tqdm=True,
                keep_pseudo_probs=True,
            )
            prob_preds.append(preds)
            masks.append(batch[1].cpu().detach().numpy())

        for thresh in tqdm(
            thresholds,
            desc=f"Optimizing Threshold with {metric} score",
            disable=disable_tqdm,
        ):
            if metric == "iou":
                iou_scores = []

                for prob_pred, mask in zip(prob_preds, masks):
                    pred = (prob_pred.detach().cpu().numpy() > thresh).astype(np.int32)
                    mask = mask.astype(np.int32)
                    iou_score = aggregated_jaccard_index(pred, mask)
                    iou_scores.append(iou_score.item())

                avg_iou = np.mean(iou_scores)

                if avg_iou > best_score:
                    best_score = avg_iou
                    best_thresh = thresh

        self.best_threshold = best_thresh

        print(
            f"Best threshold found: {self.best_threshold} with {metric} score: {best_score}"
        )

        return self.best_threshold

    def predict_segmentation(
        self,
        datamodule,
        thresh=None,
        segmenter="watershed",
        disable_tqdm=False,
        simple_min_area=500,
    ):

        thresh = thresh if thresh is not None else self.best_threshold
        out = self.predict(
            datamodule,
            thresh=thresh,
            disable_tqdm=disable_tqdm,
            keep_pseudo_probs=True,  # Use raw model output for segmentation
        ).squeeze()

        if segmenter == "watershed":
            labels = watershed_instanciation_from_pseudo_probs(
                out, threshold_abs=thresh
            )

        elif segmenter == "simple":
            labels = []
            for pred in out:
                pred = pred.numpy()
                pred = (pred > thresh).astype(np.uint8)
                pred = scipy.ndimage.binary_fill_holes(pred).astype(np.uint8)
                pred_uint8 = (pred * 255).astype(np.uint8)

                _, label = cv2.connectedComponents(pred_uint8)

                unique, counts = np.unique(label, return_counts=True)
                for u, c in zip(unique, counts):
                    if u != 0 and c < simple_min_area:
                        label[label == u] = 0

                labels.append(label)
            labels = np.array(labels)
        else:
            raise ValueError('Unknown segmenter type. Use "watershed" or "simple".')

        return labels

    def predict_segmentation_binary_masks(  # DEPRECATED
        self,
        datamodule,
        thresh=None,
        disable_tqdm=False,
        peak_local_max_footprint=np.ones((5, 5)),
        markers_structure=np.ones((3, 3)),
    ):

        thresh = thresh if thresh is not None else self.best_threshold
        preds = self.predict(
            datamodule,
            thresh=thresh,
            disable_tqdm=disable_tqdm,
        ).squeeze()
        preds_for_watershed = [pred.cpu().detach().numpy().squeeze() for pred in preds]
        labels = watershed_instanciation(
            preds_for_watershed,
            peak_local_max_footprint=peak_local_max_footprint,
            markers_structure=markers_structure,
        )
        return labels

    def on_train_end(self):
        """
        Called at the end of training to optimize the threshold on the validation set and evaluate on the test set.
        This method is only called if a validation datamodule or a test datamodule is attached to the trainer.
        """

        datamodule = self.trainer.datamodule
        if datamodule is not None and hasattr(datamodule, "val_dataloader"):
            print("Optimizing threshold on validation set at end of training...")
            self.best_threshold = self.optimize_threshold(
                datamodule, **vars(self.optimize_threshold_params)
            )
            print(f"Best threshold found: {self.best_threshold}")
            # save best threshold
            if not os.path.exists(self.trainer.checkpoint_callback.dirpath):
                os.makedirs(self.trainer.checkpoint_callback.dirpath, exist_ok=True)
            file = os.path.join(
                self.trainer.checkpoint_callback.dirpath, "best_threshold.pkl"
            )
            pickle.dump(self.best_threshold, open(file, "wb"))
        else:
            print("No validation datamodule found for threshold optimization.")
        self.best_threshold = (
            self.best_threshold if self.best_threshold is not None else 0.5
        )

        if datamodule is not None and hasattr(datamodule, "test_dataloader"):
            print("Evaluating model on test set at end of training...")
            self.eval()
            preds_seg = self.predict_segmentation(
                datamodule, thresh=self.best_threshold, disable_tqdm=False
            )
            masks = list(datamodule.test_ds.masks)
            stats = matching_dataset_dist(
                Y_trues=datamodule.test_ds.masks,
                Y_preds=preds_seg,
                thresh=2,
                use_futures=False,
            )

            iou = aggregated_jaccard_index(masks, preds_seg)
            stats["iou"] = iou
            threshold = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            ap, tp, fp, fn, f1 = average_precision(
                masks, preds_seg, threshold=threshold
            )
            stats["ap_iou"] = ap
            stats["tp_iou"] = tp
            stats["fp_iou"] = fp
            stats["fn_iou"] = fn
            stats["f1_iou"] = f1
            stats["iou_thresholds"] = threshold

            print(f"Test set results: {stats}")

            ckpt_dir = self.trainer.checkpoint_callback.dirpath

            if not os.path.exists(self.trainer.checkpoint_callback.dirpath):
                os.makedirs(self.trainer.checkpoint_callback.dirpath, exist_ok=True)
            filepath = os.path.join(ckpt_dir, "FogNet_test_set_stats.pkl")

            with open(filepath, "wb") as f:
                pickle.dump(stats, f)
            print(f"Saved test set stats to: {filepath}")

        else:
            print("No test datamodule found for evaluation at end of training.")
        return super().on_train_end()


#dataclass for optimize_threshold parameters
