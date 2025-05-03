# Copyright 2024  Kapil Devkota, Rohit Singh
# All rights reserved
# This code is available under the terms of the license available at https://github.com/rohitsinghlab/raygun

import lightning as L
import torch
import numpy as np
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import logging
from Bio.Align import substitution_matrices
import numpy as np

MINALLOWEDLENGTH = 50

class RaygunLightning(L.LightningModule):
    def __init__(self, raygun, esmalphabet, lr = 1e-3, 
                crossentropyloss = 1., 
                reconstructionloss = 1., 
                replicateloss = 1.,
                log_wandb = False, 
                save_every = 1,
                save_dir   = None):
        super().__init__()
        self.model  = raygun
        self.lr     = lr
        self.crossentropyloss = crossentropyloss
        self.reconstructloss  = reconstructionloss
        self.replicateloss    = replicateloss
        self.trainlosses      = defaultdict(list)
        self.vallosses        = defaultdict(list)
        self.epoch            = 0
        bl                    = substitution_matrices.load("BLOSUM62")
        self.blosummat        = pd.DataFrame(bl, columns = list(bl.alphabet))
        self.blosummat.index  = list(bl.alphabet)
        self.decodermodel     = raygun.esmdecoder
        self.toktoalphdict    = {k: i for i, k in esmalphabet.to_dict().items()}
        self.log_wandb        = log_wandb
        self.save_every       = save_every
        self.save_dir         = save_dir

    def wlog(self, *p): 
        if self.log_wandb:
            self.log(*p) 
        else:
            logging.info(f"{p[0]} : {p[1]}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 
                                                     start_factor = 0.5, 
                                                     end_factor = 1, total_iters = 50, 
                                                     last_epoch=-1)
        # Return optimizer and scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler" : {
                "scheduler": scheduler,
                "interval" : "step",
                "freq"     : 1
            },
        }

    def training_step(self, batch, batch_idx):
        """
        token, embedding and mask should not contain the begin and end tokens
        """
        tokens, e, mask, _ = batch
        bshape, seq_, _    = e.shape
        if mask is None:
            assert bshape == 1, "Batch is larger than 1 but no mask provided"
            ## required when replicateloss > 0
            newlengths = torch.randint(MINALLOWEDLENGTH, seq_, [1])
        else:
            lengths    = mask.sum(dim = 1)
            newlengths = torch.concat([torch.randint(MINALLOWEDLENGTH, l, [1]) 
                         for l in lengths]) 
        tloss = 0
        if self.crossentropyloss > 0:
            result, mem, crossloss = self.model(e, mask = mask, token = tokens)
            tloss                  = tloss + self.crossentropyloss * crossloss
            self.trainlosses["Cross-Entropy Loss"].append(crossloss.item())
            self.wlog("Cross-Entropy Loss", crossloss.item() if crossloss.item() < 10 else 10)
        else:
            result, mem            = self.model(e, mask = mask)
        if self.reconstructloss > 0:
            recloss                = F.mse_loss(result * mask.unsqueeze(-1), 
                                                e * mask.unsqueeze(-1))
            tloss                  = tloss + self.reconstructloss * recloss
            self.trainlosses["Reconstruction Loss"].append(recloss.item())
            self.wlog("Reconstruction Loss", recloss.item() if recloss.item() < 10 else 10)
        if self.replicateloss > 0:
            decodedemb = self.model.decode(mem, newlengths)
            reploss    = F.mse_loss(mem, self.model.encoder(decodedemb)) 
            tloss      = tloss + self.replicateloss * reploss 
            self.trainlosses["Replicate Loss"].append(reploss.item())
            self.wlog("Replicate Loss", reploss.item() if reploss.item() < 10 else 10)
        blosumv, blosumr = self.get_blosum_score(result.detach(), tokens.detach())
        self.wlog("Blosum score", blosumv)
        self.wlog("Blosum ratio", blosumr)
        return tloss

    def on_train_epoch_end(self):
        logf = f"Completed Training Epoch {self.epoch+1}: "
        for k, v in self.trainlosses.items():
            logf += f"{k} : {np.mean(v):.4f}"
        logging.info(logf)
        self.trainlosses = defaultdict(list)
        self.epoch      += 1
        if (self.epoch % self.save_every == 0) and (self.save_dir is not None):
            checkpoint_path = f"{self.save_dir}/model-{self.epoch}.sav"
            torch.save({"model_state_dict": self.model.state_dict()}, 
                      checkpoint_path)
        return

    def validation_step(self, batch, batch_idx):
        tokens, e, mask, _ = batch
        result, mem        = self.model(e, mask = mask)
        blosum_curr, blosum_curr_ratio = self.get_blosum_score(result,
                                                                tokens)
        self.wlog("Val Blosum Score", blosum_curr)
        self.wlog("Val Blosum Ratio", blosum_curr_ratio)
        self.vallosses["Blosum Score"].append(blosum_curr)
        self.vallosses["Blosum ratio"].append(blosum_curr_ratio)

    def on_validation_epoch_end(self):
        logf = f"Completed Validation Epoch {self.epoch}"
        for k, v in self.vallosses.items():
            logf += f"{k} : {np.mean(v): .4f}"
        self.validlosses = defaultdict(list)
        return

    ### Blosum scores prediction 
    def convert_tokens_to_alph(self, token, lengths):
        """
        token: tensor [batch, seqlen]
        """
        assert len(token.shape) == 2
        batch, _ = token.shape
        alphabets = []
        for i in range(batch):
            li  = lengths[i]
            tok = token[i][:li].tolist() 
            alphabets.append([self.toktoalphdict[t] for t in tok])
        return alphabets

    def get_blosum_score(self, embedding, true_token):
        """
        embedding: tensor [batch, seqlen, dim]
        true_token: tensor [batch, seqlen]
        """
        ## logging.info(f"Tokens shape {true_token.shape}, embed shape {embedding.shape}")
        batch, _, _ = embedding.shape
        lengths     = []
        
        for i in range(batch):
            tok  = true_token[i]
            lengths.append(tok[tok != 1].shape[0]) # tok being 1 implies padding
        with torch.no_grad():
            true_alph    = self.convert_tokens_to_alph(true_token.cpu().numpy(),
                                                       lengths)
            logits       = self.decodermodel(embedding)
            pred_tokens  = torch.argmax(logits, dim = -1).cpu().numpy()
            pred_alph    = self.convert_tokens_to_alph(pred_tokens, lengths)
            blcs, blrs   = [], []
            for b in range(batch):
                blc, blr       = self.compute_blosum_score(true_alph[b], 
                                                           pred_alph[b])
                blcs.append(blc)
                blrs.append(blr)
        return np.average(blcs), np.average(blrs)

    def compute_blosum_score(self, true, predicted):
        blosum_max  = 0
        blosum_curr = 0
        for p, q in zip(true, predicted):
            try:
                blosum_c_score = self.blosummat.loc[p.upper(), 
                                                    q.upper()] # if no p and q, this triggers exception
                blosum_max += self.blosummat.loc[p.upper(), 
                                                 p.upper()]
                blosum_curr += blosum_c_score
            except Exception as e:
                continue
        return blosum_curr, blosum_curr / blosum_max