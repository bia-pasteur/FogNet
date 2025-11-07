from fognet.models import FogNet
from fognet.data import FogDataModule
from lightning.pytorch.cli import LightningCLI
import torch



def main():
    torch.set_float32_matmul_precision('medium')
    LightningCLI(FogNet, FogDataModule)


if __name__ == "__main__":
    main()

#example command to run:
# conda activate fognet
# python train.py fit --config ./config/config.yaml
