"""
Training code for COMP64501 Fashion-MNIST coursework.

Run from project root with:
    uv run -m submission.fashion_training
"""

import os
import numpy as np
import torch
import torchvision

from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from submission import engine
from submission.fashion_model import Net


# -----------------------------
# Transforms (used by utils.py)
# -----------------------------


def get_transforms(mode: str = "train"):
    """
    Data preprocessing / augmentations for Fashion-MNIST.

    Only uses standard torchvision transforms (no lambdas).
    Must be deterministic in eval mode.
    """
    # These are standard-ish stats for Fashion-MNIST; the exact
    # numbers are not critical for this coursework.
    mean = (0.2860,)
    std = (0.3530,)

    if mode == "train":
        tfs = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    elif mode == "eval":
        tfs = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        # Ensure any module-like transforms are in eval mode
        for tf in tfs.transforms:
            if hasattr(tf, "train"):
                tf.eval()  # type: ignore
    else:
        raise ValueError(f"Unknown mode {mode} for transforms, must be 'train' or 'eval'.")
    return tfs


# -----------------------------
# Core training function
# -----------------------------


def train_fashion_model(
    fashion_mnist,
    n_epochs,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    USE_GPU: bool = True,
    weight_decay: float = 1e-4,
    model_variant: str = "medium",
    val_fraction: float = 0.1,
    verbose: bool = True,
    **kwargs,
):
    """
    THIS SIGNATURE MUST MATCH THE ORIGINAL SKELETON.

    Args:
        fashion_mnist: a Dataset of Fashion-MNIST samples (PIL images with transforms applied).
        n_epochs: number of training epochs.
        batch_size: minibatch size.
        learning_rate: optimizer learning rate.
        USE_GPU: whether to use GPU if available.
        weight_decay: L2 regularisation.
        model_variant: "medium" or "small" (see fashion_model.Net).
        val_fraction: fraction of training data to use for validation split.
        verbose: whether to print per-epoch stats.

    Returns:
        state_dict of the best performing model on the validation set.
    """
    # Device selection
    if USE_GPU and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if verbose:
        print(f"Using device: {device}")

    # Simple reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # If `fashion_mnist` is already a (train, val) tuple, honour that.
    if isinstance(fashion_mnist, (list, tuple)) and len(fashion_mnist) == 2:
        train_data, val_data = fashion_mnist
    else:
        # Create train-val split
        dataset_length = len(fashion_mnist)
        val_size = int(dataset_length * val_fraction)
        train_size = dataset_length - val_size
        train_data, val_data = random_split(
            fashion_mnist,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

    # Dataloaders
    train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


    # Model, loss, optimiser
    model = Net(variant=model_variant)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_state = None
    best_val_acc = 0.0

    # Training loop
    for epoch in range(1, n_epochs + 1):
        train_loss = engine.train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = engine.eval(model, val_loader, criterion, device)

        if verbose:
            print(
                f"Epoch [{epoch:02d}/{n_epochs:02d}] "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f}"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    if best_state is None:
        best_state = model.state_dict()

    return best_state


# -----------------------------
# Data loading for manual runs
# -----------------------------


def load_training_data():
    """
    Load the Fashion-MNIST training set with training transforms applied.

    This is mainly for your own training script; the marking code will do its
    own loading with get_transforms() + train_fashion_model().
    """
    print("Loading Fashion-MNIST dataset...")
    tfs = get_transforms(mode="train")

    fashion_mnist = torchvision.datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=tfs,
    )
    return fashion_mnist


# -----------------------------
# Script entry point
# -----------------------------


def main():
    """
    Convenience entry point for local training.

    - Loads Fashion-MNIST
    - Trains the model
    - Saves weights to submission/model_weights.pth
    """
    fashion_mnist = load_training_data()

    # You can tweak these for your experiments:
    n_epochs = 20
    batch_size = 128
    learning_rate = 1e-3
    weight_decay = 1e-4
    model_variant = "medium"
    USE_GPU = True

    model_weights = train_fashion_model(
        fashion_mnist,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        USE_GPU=USE_GPU,
        weight_decay=weight_decay,
        model_variant=model_variant,
        verbose=True,
    )

    os.makedirs("submission", exist_ok=True)
    model_save_path = os.path.join("submission", "model_weights.pth")
    torch.save(model_weights, f=model_save_path)
    print(f"Saved model weights to: {model_save_path}")


if __name__ == "__main__":
    main()
