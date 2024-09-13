# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import time
from argparse import ArgumentParser

import torch
import torch.nn as nn

from nas.model import ResNetSpace
from nas.data import MISO_Data_v1
import os
from torchmetrics.regression import MeanAbsolutePercentageError
from nas.darts import DartsTrainer
from torch.utils.data import DataLoader


os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

logger = logging.getLogger('nni')

def accuracy(prediction, target):

    accuracy_fn = MeanAbsolutePercentageError()

    return {"accuracy": accuracy_fn(prediction.cpu(), target.cpu())}


if __name__ == "__main__":
    parser = ArgumentParser("darts")
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--log-frequency", default=10, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--unrolled", default=False, action="store_true")
    args = parser.parse_args()

    dataset = MISO_Data_v1(26, "ref_ch4(ppm)")
    train_set, val_set = torch.utils.data.random_split(dataset, [0.7, 0.3])

    model = ResNetSpace(33)
    criterion = nn.L1Loss()

    optim = torch.optim.SGD(model.parameters(), 0.025, momentum=0.9, weight_decay=3.0E-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs, eta_min=0.001)
    trainer = DartsTrainer(
        model=model,
        loss=criterion,
        metrics=lambda output, target: accuracy(output, target),
        optimizer=optim,
        num_epochs=args.epochs,
        dataset=train_set,
        batch_size=args.batch_size,
        log_frequency=args.log_frequency,
        unrolled=args.unrolled,
        workers=0,
    )
    trainer.fit()
    final_architecture = trainer.export()
    print('Final architecture:', trainer.export())
    json.dump(trainer.export(), open('checkpoint.json', 'w'))    