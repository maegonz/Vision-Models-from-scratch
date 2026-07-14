import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from utils import plot_img_mask
from typing import Union
from copyreg import pickle

import time
from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity


def training(model: nn.Module,
            train_loader: DataLoader,
            criterion: nn.Module,
            optimizer: Optimizer,
            device: torch.device,
            epochs: int,
            val_loader: DataLoader=None,
            use_amp: bool=True,
            save_model: Union[bool, str]=False):
    """
    Train a PyTorch model with optional Automatic Mixed Precision.

    Parameters
    ----------
    model : nn.Module
        The neural network model to be trained.
    train_loader : DataLoader
        DataLoader providing the training dataset.
    criterion : nn.Module
        Loss function used to compute training loss.
    optimizer : torch.optim.Optimizer
        Optimizer used to update model parameters.
    device : torch.device
        Device on which to train the model ('cpu' or 'cuda').
    epochs : int
        Number of training epochs.
    val_loader : DataLoader, optional
        DataLoader providing the validation dataset.
        If None, no validation is performed. Default is None.
    use_amp : bool, optional
        Whether to use AMP.
        AMP is enabled only when using a CUDA device. Default is True.

    Returns
    -------
    train_losses : list of float
        Average training loss for each epoch.
    train_accuracies : list of float
        Training accuracy (percentage) for each epoch.
    val_losses : list of float
        Validation loss for each epoch.
        Empty if val_loader is None.
    val_accuracies : list of float
        Validation accuracy (percentage) for each epoch.
        Empty if val_loader is None.
    """

    model.to(device)
    # model = torch.compile(model)

    use_amp = use_amp and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    epoch_tqdm = tqdm(range(epochs), desc="Training Progress")

    for epoch in epoch_tqdm:
        model.train()
        running_loss = torch.tensor(0.0, device=device)
        correct_pixels = torch.tensor(0, device=device)
        total_pixels = torch.tensor(0, device=device)


        for j, (imgs, masks) in enumerate(train_loader):

            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            
            with autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
                outputs = model(imgs)
                loss = criterion(outputs, masks)

            optimizer.zero_grad(set_to_none=True)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.detach() * imgs.size(0)
            predicted = torch.argmax(outputs, dim=1)
            correct_pixels += (predicted == masks).sum().detach()
            total_pixels += masks.numel()
            
        epoch_loss = running_loss.item() / len(train_loader.dataset)
        epoch_acc = 100.0 * correct_pixels.item() / total_pixels.item()

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        if val_loader is not None:

            val_loss, val_accuracy = evaluating(
                model, val_loader, criterion, device, use_amp=use_amp, plot_output=False
            )
            
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            # if val_accuracy > max(val_accuracies[:-1], default=0) and save_model:
            save_path = save_model if isinstance(save_model, str) else f"./unet/saved/best_models/unet_camvid_best_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), save_path)

            epoch_tqdm.set_postfix(
                train_loss=epoch_loss,
                val_loss=val_loss,
            )
        else:
            epoch_tqdm.set_postfix(train_loss=epoch_loss)
    
    np.save("./unet/saved/metrics/train_losses.npy", train_losses)
    np.save("./unet/saved/metrics/train_accuracies.npy", train_accuracies)
    np.save("./unet/saved/metrics/val_losses.npy", val_losses)
    np.save("./unet/saved/metrics/val_accuracies.npy", val_accuracies)

    return train_losses, train_accuracies, val_losses, val_accuracies

        
def evaluating(model: nn.Module, 
               data_loader: DataLoader,
               criterion: nn.Module,
               device: torch.device,
               plot_output: bool = True,
               save_path: str = "./unet/saved/outputs/",
               use_amp: bool = True):
    """
    Evaluate a PyTorch model on a dataset with optional AMP.

    Parameters
    ----------
    model : nn.Module
        The trained model to be evaluated.
    data_loader : DataLoader
        DataLoader providing the evaluation dataset.
    criterion : nn.Module
        Loss function used to compute evaluation loss.
    device : torch.device
        Device on which evaluation is performed ('cpu' or 'cuda').
    use_amp : bool, optional
        Whether to use Automatic Mixed Precision (AMP).
        AMP is enabled only when using a CUDA device. Default is True.

    Returns
    -------
    avg_loss : float
        Average loss over the entire dataset.
    avg_accuracy : float
        Average accuracy (percentage) over the entire dataset.
    """

    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    correct_pixels = torch.tensor(0, device=device)
    total_pixels = torch.tensor(0, device=device)

    with torch.no_grad():
        for k, (imgs, masks) in enumerate(data_loader):
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=use_amp):
                outputs = model(imgs)
                loss = criterion(outputs, masks)

            total_loss += loss.detach() * imgs.size(0)
            predicted = torch.argmax(outputs, dim=1)
            correct_pixels += (predicted == masks).sum().detach()
            total_pixels += masks.numel()

            if k == 2 and plot_output:
                img_plot, mask_plot, pred_plot = imgs[k].cpu(), masks[k].cpu(), predicted[k].cpu()
                plot_img_mask(img_plot, mask_plot, pred_plot, save_path=save_path, show=False)

            del outputs, loss, imgs, masks

        # torch.cuda.empty_cache()

    avg_loss = total_loss.item() / len(data_loader.dataset)
    avg_accuracy = 100.0 * correct_pixels.item() / total_pixels.item()
    return avg_loss, avg_accuracy


# def prediction(model: nn.Module, 
#                data_loader: DataLoader,
#                device: torch.device,
#                cm: bool = True,):
#     """
#     Evaluate a PyTorch model on a dataset with optional AMP.

#     Parameters
#     ----------
#     model : nn.Module
#         The trained model to be evaluated.
#     data_loader : DataLoader
#         DataLoader providing the evaluation dataset.
#     device : torch.device
#         Device on which evaluation is performed ('cpu' or 'cuda').
#     cm : bool, optional
#         Whether to display the confusion matrix.
#         Default is True.

#     Returns
#     -------
#     all_preds : list[float]
#         All predicted masks over the entire dataset.
#     all_masks : list[float]
#         All true masks over the entire dataset.
#     """
#     model.eval()
#     all_preds = []
#     all_masks = []
#     with torch.no_grad():
#         for imgs, masks in data_loader:
#             imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
#             outputs = model(imgs)
#             _, predicted = torch.max(outputs, 1)
#             all_preds.extend(predicted.cpu().numpy())
#             all_masks.extend(masks.cpu().numpy())
    
#     if cm:
#         cm = confusion_matrix(all_masks, all_preds)
#         sns.heatmap(cm, fmt='d', cmap='YlGnBu')
#         plt.xlabel('Predicted')
#         plt.ylabel('True Label')
#         plt.title('Confusion Matrix')
#         plt.show()
    
#     return all_preds, all_masks