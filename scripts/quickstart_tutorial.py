import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from constants import SAVED_MODELS_DIR

# TODO probably move this to a jupyter notebook for iteration.

EPOCHS = 5
BATCH_SIZE = 64


DEVICE = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
MODEL_OUTPUT_FILENAME = "quickstart_tutorial_model.pth"

# TODO NEXT STEPS: Continue tutorials here: https://docs.pytorch.org/tutorials/beginner/basics/intro.html


def create_loaders(batch_size: int):
    # Train / test downloads from open datasets
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_dat = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    # Data Loaders are basically iterable wrappers; have helpers for batching, shuffling, etc.
    train_loader = DataLoader(
        training_data,
        batch_size=batch_size,
    )
    test_loader = DataLoader(
        test_dat,
        batch_size=batch_size,
    )

    print("TEST LOADER")
    print("=" * 20)
    for X, y in test_loader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    return train_loader, test_loader


class MyFirstNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),  # Flatten the 28x28 images to a vector of size 784
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),  # Output layer for 10 classes
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    @classmethod
    def create(cls):
        return cls().to(DEVICE)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    # What does this train do versus below?
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)

        # Prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            # Print loss every 100 batches
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():  # nice
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            test_loss = loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \nAccuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


def main():
    print(f"Using {DEVICE} device")
    model = MyFirstNN.create()
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    train_loader, test_loader = create_loaders(BATCH_SIZE)

    for t in range(EPOCHS):
        print(f"\nEpoch {t + 1}\n" + "=" * 20 + "\n")
        train(train_loader, model, loss_fn, optimizer)
        test(test_loader, model, loss_fn)

    print("DONE! Saving model...")

    output_filename = f"{SAVED_MODELS_DIR}{MODEL_OUTPUT_FILENAME}"
    torch.save(model.state_dict(), output_filename)

    print(f"Model saved to {output_filename}")


if __name__ == "__main__":
    main()
