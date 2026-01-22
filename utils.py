import matplotlib.pyplot as plt
import seaborn as sns

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