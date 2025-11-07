import torch
import torch.nn as nn
import lightning as L
import os
import pickle
from torch.nn import functional as F
from tqdm import tqdm
from fognet.utils import watershed_instanciation
import numpy as np
import cv2
from fognet.metrics import aggregated_jaccard_index


class UNetBlock(nn.Module):
    """
    A single block of the U-Net architecture, consisting of two convolutional layers
    followed by ReLU activations. This block is typically used in the encoder and decoder parts of the U-Net.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetBase(nn.Module):
    """
    A U-Net model for image segmentation tasks. It consists of an encoder, a bottleneck,
    and a decoder with skip connections.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        features (list): List of integers representing the number of features in each encoder block.
    """

    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        prev_channels = in_channels
        for feat in features:
            self.encoder.append(UNetBlock(prev_channels, feat))
            prev_channels = feat

        self.bottleneck = UNetBlock(prev_channels, prev_channels * 2)
        prev_channels = prev_channels * 2

        self.up_transpose = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for feat in reversed(features):
            self.up_transpose.append(
                nn.ConvTranspose2d(prev_channels, feat, kernel_size=2, stride=2)
            )
            self.decoder.append(UNetBlock(prev_channels, feat))
            prev_channels = feat

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        print(
            f"Initialized UNet with in_channels={in_channels}, out_channels={out_channels}"
        )

    def forward(self, x):
        skip_connections = []

        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(len(self.up_transpose)):
            x = self.up_transpose[idx](x)
            skip = skip_connections[idx]

            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])

            x = torch.cat((skip, x), dim=1)
            x = self.decoder[idx](x)

        return self.final_conv(x)

    def predict_step(self, batch, batch_idx):
        x = batch[0]
        return self(x)


class UNet(L.LightningModule):
    """
    A PyTorch Lightning module for training a U-Net model for image segmentation tasks.
    Args:
        in_channels (int): Number of input channels (e.g., 1 for grayscale images).
        out_channels (int): Number of output channels (e.g., 1 for binary segmentation).
        lr (float): Learning rate for the optimizer.
    """

    def __init__(self, in_channels=1, out_channels=1, lr=1e-3, max_epochs=30):

        self.save_hyperparameters()

        super().__init__()
        self.model = UNetBase(in_channels, out_channels)
        self.lr = lr
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_epochs = max_epochs
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self._compute_loss(logits, x, y)

        # Logging
        self.log("train_loss", loss)
        opt = self.trainer.optimizers[0]
        lr = opt.param_groups[0]["lr"]
        self.log("lr", lr, on_step=True, on_epoch=True, prog_bar=True)

        self.log("train/loss", loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self._compute_loss(logits, x, y)
        self.log("val_loss", loss, prog_bar=True)
        preds = torch.sigmoid(logits)

        preds = (preds > 0.5).float().cpu().detach().numpy()
        y = (y > 0.5).float().cpu().detach().numpy()
        preds = preds.astype(np.int32)
        y = y.astype(np.int32)
        iou = float(aggregated_jaccard_index([preds], [y]))

        self.log("val_iou", iou, prog_bar=True)

        return loss

    def _compute_loss(self, logits, x, y):
        y = (y > 0).to(torch.float32).unsqueeze(1)  # Ensure y has shape (B, 1, H, W)
        return self.loss_fn(logits, y)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # optimizer=torch.optim.SGD(self.parameters(), lr=self.lr)

        # Warmup scheduler: linearly increase LR for the first 5 epochs
        warmup_epochs = int(self.max_epochs / 6)

        def warmup_lambda(epoch):
            return min(1.0, (epoch + 1) / warmup_epochs)

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=warmup_lambda
        )

        # Main scheduler: StepLR after warmup
        main_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(self.max_epochs / 3), gamma=0.1
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_after_backward(self):
        # Log the global norm of all gradients
        total_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        self.log("train/grad_norm", total_norm, on_step=True, prog_bar=False)

    def predict_from_dataloader(
        self, dataloader, thresh=0.5, disable_tqdm=False, keep_pseudo_probs=False
    ):
        """
        Predicts segmentation masks from a given dataloader. Used mostly for evaluation, and in the final predict method.

        Args:
            dataloader (DataLoader): The dataloader containing the data to predict on.
            thresh (float): Threshold for binary segmentation.

        Returns:
            torch.Tensor or tuple: Predictions as a tensor
        """
        predictions = []
        self.eval()
        for batch in tqdm(dataloader, desc="Predicting", disable=disable_tqdm):

            x = batch[0]
            x = x.to(self.device)
            preds = self(x)
            preds = torch.sigmoid(preds)
            if not keep_pseudo_probs:

                # save pred in pkl
                # pickle.dump(preds.cpu().detach().numpy(), open("preds.pkl", "wb"))
                preds = (preds > thresh).float()
            else:
                # If keep_pseudo_probs is True, we return the raw model output
                preds = preds.cpu().detach()
            predictions.append(preds)
        predictions = torch.cat(predictions, dim=0)

        return predictions

    def optimize_threshold(
        self,
        datamodule,
        min_thresh=0.2,
        max_thresh=0.8,
        step=0.1,
        metric="iou",
        val_set_size=5,
        disable_tqdm=False,
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

                    mask = (mask > 0).astype(np.int32)

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

    def predict(
        self, datamodule, thresh=None, disable_tqdm=False, keep_pseudo_probs=False
    ):
        """
        Predicts segmentation masks using the model on the provided data module. Optimizes the threshold on valuidation data if thresh is None.
        Predictions are performed on the prediction set of the datamodule, which is the same as the test set.


        Args:
            datamodule (SegmentationDataModule): The data module containing the data to predict on.
            thresh (float, optional): Threshold for binary segmentation. If None, the threshold will be optimized using the validation set.

        Returns:
            torch.Tensor or tuple: Predictions as a tensor.
        """
        thresh = self.best_threshold if thresh is None else thresh
        if thresh is None:

            if datamodule.val_dataloader() is None and thresh is None:
                raise ValueError(
                    "Validation dataloader is not available. Cannot optimize threshold without validation data."
                )
            print(
                "No threshold provided, optimizing threshold using validation data..."
            )

            thresh = self.optimize_threshold(
                datamodule,
                min_thresh=0.2,
                max_thresh=0.8,
                step=0.1,
                metric="iou",
                val_set_size=5,
                disable_tqdm=disable_tqdm,
            )
            self.log("optimized_threshold", thresh)
            print(f"Optimized threshold: {thresh}")

        preds = self.predict_from_dataloader(
            datamodule.test_dataloader(),
            thresh=thresh,
            disable_tqdm=disable_tqdm,
            keep_pseudo_probs=keep_pseudo_probs,
        )
        return preds

    def predict_segmentation(
        self,
        datamodule,
        thresh=None,
        segmenter="watershed",
        disable_tqdm=False,
        peak_local_max_footprint=np.ones((5, 5)),
        markers_structure=np.ones((3, 3)),
    ):

        preds = self.predict(
            datamodule, thresh=thresh, disable_tqdm=disable_tqdm
        ).squeeze()
        if self.hparams.out_channels == 1:
            preds_for_watershed = [
                pred.cpu().detach().numpy().squeeze() for pred in preds
            ]
        else:
            preds_for_watershed = [
                pred.cpu().detach().numpy()
                for pred in preds.reshape(
                    len(preds) * self.hparams.out_channels, *preds.shape[2:]
                )
            ]  # Convert to numpy arrays the inidivual preds
            preds_for_watershed = preds_for_watershed[
                self.hparams.out_channels // 2 :: self.hparams.out_channels
            ]  # Take only the first channel of each prediction, which is the segmentation mask

        if segmenter == "watershed":
            labels = watershed_instanciation(
                preds_for_watershed,
                peak_local_max_footprint=peak_local_max_footprint,
                markers_structure=markers_structure,
            )
        elif segmenter == "simple":
            labels = []
            for pred in preds_for_watershed:
                pred_uint8 = (pred * 255).astype(np.uint8)
                _, label = cv2.connectedComponents(pred_uint8)

                # remove labels smaller than min_area
                min_area = 500
                unique, counts = np.unique(label, return_counts=True)
                for u, c in zip(unique, counts):
                    if u != 0 and c < min_area:
                        label[label == u] = 0

                labels.append(label)
        else:
            raise ValueError(f"Unknown segmenter: {segmenter}")

        return labels

    def load_best_threshold(self, checkpoint_path):
        """
        Load model weights from a checkpoint file.

        Args:
            checkpoint_path (str): Path to the checkpoint file.

        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint["state_dict"])

        # find folder in which the checkpoint is located
        checkpoint_dir = os.path.dirname(checkpoint_path)
        print(f"Model weights loaded from {checkpoint_path}")
        if os.path.exists(os.path.join(checkpoint_dir, "unet_best_threshold.pkl")):
            self.best_threshold = pickle.load(
                open(os.path.join(checkpoint_dir, "unet_best_threshold.pkl"), "rb")
            )
            print(f"Loaded best threshold: {self.best_threshold}")
        else:
            self.best_threshold = None
            print("No best threshold file found")
        print(f"Model weights loaded from {checkpoint_path}")
