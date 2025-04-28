import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import pickle
from torch.utils.data import DataLoader, Dataset
import glob2
from functools import reduce
import datetime
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils.dataset_class import LesionDataset3D
from utils import dataset_perpaziente as dp
from utils.collate_fn import custom_collate_fn
from utils.model import *
from time import time 
from torch.optim.lr_scheduler import StepLR
import json
from loss import *
from utils.training_class import Trainer

# Declare device as a global variable
global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def main(gridsize=(16, 128, 128)):
    print(f"Device in use: {device}")

    root_path = "/srv/LongTerm01/pcaterino/Duke/Custom_Dataset"
    auxiliary_data_root_path = "/home/pcaterino/duke_aux/"
    auxiliary_data_file_names = ["dataset DUKE training.ods", "BCS-DBT-boxes-train-v2.csv", "SELECTED PATIENTS.ods"]


    # Prepare pandas dataframes
    classe_dati = dp.DataProcessor(root_path,auxiliary_data_root_path )
    classe_dati.preprocess_aux_files()
    df_tot, df_paths = classe_dati.load_data_paths()
    df = classe_dati.filter_data(["clips"])
    # Create dataset and dataloader
    train_dataset = LesionDataset3D("/home/pcaterino/3d_effort/resized/2025-03-25_13-08/3/train", df)
    val_dataset = LesionDataset3D("/home/pcaterino/3d_effort/resized/2025-03-25_13-08/3/val", df,split_type="val")
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=6, pin_memory=True,collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=6, pin_memory=True,collate_fn=custom_collate_fn)
    pos_weight_train = train_dataset.pos_weights
    pos_weight_val = val_dataset.pos_weights
    print(f"Pos weight train: {pos_weight_train}")
    print(f"Pos weight val: {pos_weight_val}")
    # Initialize the model
    model = DenseYOLO3D(img_channels=1,out_channels= 7) #TumorClassifier3D() #DenseYOLO3D(img_channels=1)
    # loss
    loss = YOLO3DLoss(alpha=0.5, gamma=2.0, lambda_coord=5.0)
    #optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, min_lr=1e-6, threshold=0.1)

    trainer = Trainer(model, train_dataloader, val_dataloader, loss, optimizer, scheduler, device)
    path= "/home/pcaterino/DenseYolo3D/model_data/"


    a= time()

    # Check the output of the dataset class to verify the shape of the elements
    for idx in range(len(train_dataset)):
        sample = train_dataset[idx]
        print(f"Sample {idx + 1}:")
        print(f"Input shape: {sample[0].shape}")
        print(f"Target shape: {sample[1].shape}")
        if idx == 2:  # Limit the output to the first 3 samples
            break

    # Check one loop of the dataloader to verify shapes
    for batch_idx, (inputs, targets,shapes) in enumerate(train_dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f"Inputs shape: {inputs.shape}")
        print(f"Targets shape: {targets.shape}")
        print(f"Shapes: {shapes}")
        # Exit after 2 loop
        if idx == 2:  # Limit the output to the first 3 samples
            break

    trainer.train(num_epochs=100,path = path, min_improvement=1)
    b=time()
    print((b-a)/60)




if __name__ == "__main__":
    main()


