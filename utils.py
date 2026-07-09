import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_metrics(
    train_losses1, val_losses1, train_accs1, val_accs1,
    epochs,
    train_losses2=None, val_losses2=None, train_accs2=None, val_accs2=None,
    opt1_name="Optimizer 1",
    opt2_name="Optimizer 2"
):
    sns.set_theme(style="whitegrid", context="talk")
    epochs_range = range(1, epochs + 1)

    # ---- Check if second optimizer is provided ----
    two_opts = all(v is not None for v in
                   [train_losses2, val_losses2, train_accs2, val_accs2])

    if two_opts:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)

        # Optimizer 1 - Loss
        axes[0, 0].plot(epochs_range, train_losses1, label="Train Loss")
        axes[0, 0].plot(epochs_range, val_losses1, label="Val Loss")
        axes[0, 0].set_title(f"{opt1_name} - Loss")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()

        # Optimizer 2 - Loss
        axes[0, 1].plot(epochs_range, train_losses2, label="Train Loss")
        axes[0, 1].plot(epochs_range, val_losses2, label="Val Loss")
        axes[0, 1].set_title(f"{opt2_name} - Loss")
        axes[0, 1].legend()

        # Optimizer 1 - Accuracy
        axes[1, 0].plot(epochs_range, train_accs1, label="Train Accuracy")
        axes[1, 0].plot(epochs_range, val_accs1, label="Val Accuracy")
        axes[1, 0].set_title(f"{opt1_name} - Accuracy")
        axes[1, 0].set_xlabel("Epochs")
        axes[1, 0].set_ylabel("Accuracy")
        axes[1, 0].legend()

        # Optimizer 2 - Accuracy
        axes[1, 1].plot(epochs_range, train_accs2, label="Train Accuracy")
        axes[1, 1].plot(epochs_range, val_accs2, label="Val Accuracy")
        axes[1, 1].set_title(f"{opt2_name} - Accuracy")
        axes[1, 1].set_xlabel("Epochs")
        axes[1, 1].legend()

    else:
        # ---- Single optimizer: 1x2 layout ----
        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)

        # Loss
        axes[0].plot(epochs_range, train_losses1, label="Train Loss")
        axes[0].plot(epochs_range, val_losses1, label="Val Loss")
        axes[0].set_title(f"{opt1_name} - Loss")
        axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        # Accuracy
        axes[1].plot(epochs_range, train_accs1, label="Train Accuracy")
        axes[1].plot(epochs_range, val_accs1, label="Val Accuracy")
        axes[1].set_title(f"{opt1_name} - Accuracy")
        axes[1].set_xlabel("Epochs")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()

    plt.tight_layout()
    plt.show()


from typing import List, Optional, Union

def plot_img_mask(img: Union[torch.tensor, List[torch.tensor]], 
                  mask: Union[torch.tensor, List[torch.tensor]], 
                  pred_mask: Union[torch.tensor, List[torch.tensor]]=None, 
                  num_samples: int=5,
                  class_names: List[str]=None):
    """
    Plot an input image with its ground truth mask and optional prediction.

    Parameters
    ----------
    img : torch.Tensor or list[torch.Tensor]
        Image tensor in ``(C, H, W)`` format.
    mask : torch.Tensor or list[torch.Tensor]
        Ground truth mask tensor in ``(H, W)`` format.
    pred_mask : torch.Tensor or list[torch.Tensor], optional
        Predicted mask tensor in ``(H, W)`` format. Default is ``None``.
    num_samples : int, optional
        Number of samples to plot. Default is ``5``.
    class_names : list[str], optional
        Class names corresponding to mask values. Default is ``None``.
    """

    if isinstance(img, list):
        img = img[:num_samples]
    if isinstance(mask, list):
        mask = mask[:num_samples]
    if pred_mask is not None and isinstance(pred_mask, list):
        pred_mask = pred_mask[:num_samples]


    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3 if pred_mask is not None else 2, 1)
    plt.imshow(img)
    plt.title("Input Image")
    plt.axis('off')

    plt.subplot(1, 3 if pred_mask is not None else 2, 2)
    plt.imshow(mask, alpha=0.5)
    plt.title("Ground Truth Mask")
    plt.axis('off')

    if pred_mask is not None:
        pred_mask = pred_mask
        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask, alpha=0.5)
        plt.title("Predicted Mask")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def unnormalize_image(img: torch.Tensor, 
                      mean: List[float]=[0.485,0.456,0.406], 
                      std: List[float]=[0.229,0.224,0.225]):
    """
    Unnormalize an image tensor.

    Parameters
    ----------
    img : torch.Tensor
        Image tensor in ``(C, H, W)`` format.
    mean : list[float]
        Mean values for each channel.
    std : list[float]
        Standard deviation values for each channel.

    Returns
    -------
    torch.Tensor
        Unnormalized image tensor in ``(C, H, W)`` format.
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    
    unnormalized_img = img * std + mean
    return unnormalized_img