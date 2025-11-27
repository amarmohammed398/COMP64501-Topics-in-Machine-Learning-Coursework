"""
Training utilities for Fashion-MNIST.

The training function must keep its name and signature because it will be
called automatically during marking. You are allowed to edit the internal
logic, defaults, comments, and structure.

This script expects:
- submission/fashion_model.py containing class Net
- submission/engine.py providing train() and eval()
"""

import os
import numpy as np
import torch
import torchvision

from submission import engine
from submission.fashion_model import Net


def train_fashion_model(
    fashion_mnist,
    n_epochs,
    batch_size=4,
    learning_rate=0.001,
    USE_GPU=False,
):
    """
    Train a model on Fashion-MNIST.

    Arguments and return value MUST remain unchanged for marking.

    Parameters:
        fashion_mnist: Dataset
        n_epochs: training epochs
        batch_size: dataloader batch size
        learning_rate: optimiser LR
        USE_GPU: optional GPU use

    Returns:
        model.state_dict(): trained weights
    """

    # ---- Device selection ----
    if USE_GPU and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ---- Dataset split (80/20 train/val) ----
    train_size = int(0.8 * len(fashion_mnist))
    val_size = len(fashion_mnist) - train_size
    train_data, val_data = torch.utils.data.random_split(
        fashion_mnist, [train_size, val_size]
    )

    # ---- DataLoaders ----
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
    )

    # ---- Model, loss, optimizer ----
    model = Net().to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Optional LR schedule
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=7,
        gamma=0.5,
    )

    # ---- Training loop ----
    for epoch in range(n_epochs):
        train_loss = engine.train(model, train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch + 1}/{n_epochs}], Training Loss: {train_loss:.4f}")

        val_loss, accuracy = engine.eval(model, val_loader, criterion, device)
        print(f"Epoch [{epoch + 1}/{n_epochs}], Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")

        scheduler.step()

    # Required by marking script — do not modify
    return model.state_dict()


def get_transforms(mode="train"):
    """
    Construct transforms for training or evaluation.

    Restrictions:
    - Only torchvision transforms allowed.
    - No lambda transforms.
    - Must remain deterministic in eval mode.
    """

    if mode == "train":
        tfs = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ])

    elif mode == "eval":
        tfs = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ])

        # Ensure deterministic behaviour
        for tf in tfs.transforms:
            if hasattr(tf, "train"):
                tf.eval()

    else:
        raise ValueError("Mode must be 'train' or 'eval'.")

    return tfs


def load_training_data():
    """
    Load Fashion-MNIST (train split only).
    Applies transforms defined above.
    """

    print("Loading Fashion-MNIST dataset...")
    fashion_mnist = torchvision.datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
    )

    # Attach preprocessing pipeline
    fashion_mnist.transform = get_transforms(mode="train")
    return fashion_mnist


def main():
    """
    Example training entry point.
    You may expand this for hyperparameter search,
    cross-validation, etc.
    """

    fashion_mnist = load_training_data()

    model_weights = train_fashion_model(
        fashion_mnist,
        n_epochs=20,
        batch_size=64,
        learning_rate=0.001,
    )

    # Save trained weights (required for submission)
    model_save_path = os.path.join("submission", "model_weights.pth")
    torch.save(model_weights, model_save_path)


if __name__ == "__main__":
    main()