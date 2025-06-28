from Data.data_loader import DataPrepare,dataloader
from Data.data import load_mat_dataset,image_with_label_list
from model import Attention,Encoder,Decoder,Draw
from loss import loss

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

IMG_HEIGHT = 32
IMG_WIDTH = 32
IMG_CHANNELS = 3

ENC_HIDDEN_DIM = 800
DEC_HIDDEN_DIM = 800
LATENT_DIM = 100      
T = 10 
N = 5  

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
LOG_INTERVAL = 100
KL_ANNEAL_EPOCHS  = 5 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = r"C:\Users\Kusha\OneDrive\Desktop\LLM\DRAW\Data\dataset\train_32x32.mat"
IMAGE_KEY = 'images'
LABEL_KEY = 'labels'

X,labels = load_mat_dataset(DATA_PATH)
data_list = image_with_label_list(X, labels)

loader = dataloader(data_list, batch_size=32, shuffle=True, num_workers=0)

def train_draw_model():
    model = Draw(
        img_width=IMG_WIDTH,
        img_height=IMG_HEIGHT,
        img_channels=IMG_CHANNELS,
        enc_hidden_dim=ENC_HIDDEN_DIM,
        dec_hidden_dim=DEC_HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        T=T,
        N=N
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_Lx = 0.0
        running_Lz = 0.0

        beta = min(1.0, epoch / KL_ANNEAL_EPOCHS)

        for batch_idx, (images, _) in enumerate(loader, start=1):
            images = images.to(DEVICE).float() / 255.0

            optimizer.zero_grad()
            raw_canvases, mu_list, sigma_list = model(images)

            total_loss, Lx, Lz = loss(
                images,
                raw_canvases,
                mu_list,
                sigma_list
            )
            total_loss.backward()
            optimizer.step()

            running_Lx += Lx.item()
            running_Lz += Lz.item()

            if batch_idx % LOG_INTERVAL == 0:
                print(f"Epoch[{epoch}/{NUM_EPOCHS}] "
                      f"Batch[{batch_idx}/{len(loader)}]  "
                      f"β={beta:.2f}  "
                      f"Lx: {Lx.item():.4f}  Lz: {Lz.item():.4f}  Total: {total_loss.item():.4f}")

        avg_Lx = running_Lx / len(loader)
        avg_Lz = running_Lz / len(loader)
        print(f"--- Epoch {epoch} Complete ---")
        print(f"Avg Lx (bits/pixel): {avg_Lx:.4f}  "
              f"Avg Lz (nats/image): {avg_Lz:.4f}  β={beta:.2f}\n")


if __name__ == '__main__':
    train_draw_model()