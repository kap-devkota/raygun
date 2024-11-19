# Copyright 2024  Kapil Devkota, Rohit Singh
# All rights reserved
# This code is available under the terms of the license available at https://github.com/rohitsinghlab/raygun
import argparse
import sys
sys.path.append("src/")
from b_raygun.model.raygun import Raygun 
from b_raygun.model.esmdecoder import DecoderBlock
from b_raygun.loader import RaygunData
from b_raygun.train_utils import train
from b_raygun.model.ltraygun import RaygunLightning
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import esm
import os
import pandas as pd
import itertools
import time
import json
from Bio.Align import substitution_matrices
import subprocess
import logging 
import lightning as L
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger

torch.set_float32_matmul_precision('high')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

@hydra.main(config_path="configs/", config_name="train",
            version_base = None)
def main(config: DictConfig):
    logger.info("Running Raygun training/finetuning...")
    config = OmegaConf.to_container(config, resolve=True)

    # create model and embedding folders
    os.makedirs(config["model_saveloc"], exist_ok = True)
    if config["esm2_embedding_saveloc"] is not None:
        os.makedirs(config["model_saveloc"], exist_ok = True)

    # Use ESM-2 650M
    esmmodel, esmalphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esmmodel              = esmmodel.to(0)
    esmmodel.eval()

    if not config["debug_mode"]:
        wandb_logger = WandbLogger(project = "Batched Training Raygun")
    else:
        wandb_logger = None

    logger.info(f"Using pre-trained checkpoint.")
    _, _, hyparams = torch.hub.load('rohitsinghlab/raygun', 
                                    'pretrained_uniref50_95000_750M')
    model          = Raygun(numencoders = config["numencoders"],
                            numdecoders = config["numdecoders"],
                            esmdecodertotokenfile = "data/models/esm-decoder.sav",
                            esm_alphabet = esmalphabet.to_dict())
    rayltmodule    = RaygunLightning(model, 
                                     esmalphabet,
                                     lr = config["lr"],
                                     log_wandb = not config["debug_mode"])         
    ## train and validation loaders
    traindata = RaygunData(fastafile = config["trainfasta"],
                           alphabet  = esmalphabet,
                           model     = esmmodel, 
                           device    = 0)
    trainloader = DataLoader(traindata, 
                             shuffle = False, 
                             batch_size = config["batch_size"],
                             collate_fn = traindata.collatefn)
    validdata = RaygunData(fastafile = config["validfasta"],
                           alphabet  = esmalphabet,
                           model     = esmmodel,
                           device    = 0)
    validloader = DataLoader(validdata, 
                            shuffle = False,
                            batch_size = config["batch_size"], 
                            collate_fn = validdata.collatefn)
    # Start the training
    trainer = L.Trainer(logger = wandb_logger, accelerator="gpu", 
                        devices=config["devices"], strategy="ddp",
                        max_epochs=config["epoch"])
    trainer.fit(rayltmodule, trainloader, 
                validloader)
    return 

if __name__ == "__main__":
    main()
    
