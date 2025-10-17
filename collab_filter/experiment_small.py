"""
Goal here is to get a proper collab filtering model runnning on MovieLens 100k.
Mostly looking for comfort with pytorch syntax, workflow, tools etc.
"""

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os


print(f"Base path: {os.getcwd()}")

OUT_DIR = "collab_filter/out/experiment_small"
RATINGS_CSV_PATH = "collab_filter/data/ml-latest-small/ratings.csv"

DEVICE = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

print(f"Using DEVICE={DEVICE}")



MODEL_OUTPUT_FILENAME = "collab_filter_model.pth"
METRICS_OUTPUT_FILENAME = "collab_filter_metrics.json"


# TODO - hyperparams

TRAIN_BATCH_SIZE = 256
EPOCHS = 50
MODEL_DIMS = 32
LR = 1e-3

criterion = torch.nn.MSELoss()


class RatingsDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        # Numpy arrs w/ explicit datatype conversions
        self.users = df["userId"].values.astype('int64')
        self.items = df["movieId"].values.astype('int64')
        self.ratings = df["rating"].values.astype('float32') 

    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return {
            "user": self.users[idx],
            "item": self.items[idx],
            "rating": self.ratings[idx],
        }

print("Reading data...")
df = pd.read_csv(RATINGS_CSV_PATH)


print("Mapping idxs...")
user_to_idx = {user_id: idx for idx, user_id in enumerate(df["userId"].unique())}
item_to_idx = {item_id: idx for idx, item_id in enumerate(df["movieId"].unique())}

# Remap to unique contiguous idxs
df["userId"] = df["userId"].map(user_to_idx)
df["movieId"] = df["movieId"].map(item_to_idx)

n_users = len(user_to_idx)
n_items = len(item_to_idx)



train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = RatingsDataset(train_df)
test_dataset = RatingsDataset(test_df)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=TRAIN_BATCH_SIZE)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=512)


# Next up - let's build a simple collaborative filtering model
class CollaborativeFilteringModel(torch.nn.Module):
    def __init__(self, n_users, n_items, embedding_dim):
        super(CollaborativeFilteringModel, self).__init__()
        self.user_embedding = torch.nn.Embedding(n_users, embedding_dim)
        self.item_embedding = torch.nn.Embedding(n_items, embedding_dim)

    # You pass in a user and item to predict over.
    # Some thoughts;
    # For new items/users getting added, you basically need to initialize them with some average vector (cold start)
    # For online learning, I wonder if folks tend to train a new model each time, or 
    def forward(self, user, item):
        user_vec = self.user_embedding(user)
        item_vec = self.item_embedding(item)
        return (user_vec * item_vec).sum(dim=1)

print("Initializing model...")
model = CollaborativeFilteringModel(n_users, n_items, MODEL_DIMS).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Train
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}\n" + "=" * 20 + "\n")

    for batch_idx, batch in enumerate(train_dataloader):
        user = batch["user"].to(DEVICE)
        item = batch["item"].to(DEVICE)
        rating = batch["rating"].to(DEVICE)

        # Fwd
        pred = model(user, item)
        loss = criterion(pred, rating.float())


        # Bwd
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print(f"Train Batch {batch_idx}, Loss: {loss.item():.4f}")

    print(f"Train Final loss: {loss.item():.4f}")


# Test
model.eval()  # Set model to evaluation mode (disables dropout, etc.)

test_loss = 0.0
num_batches = 0

with torch.no_grad():  # Disable gradient computation for efficiency
    for batch in test_dataloader:
        user = batch["user"].to(DEVICE)
        item = batch["item"].to(DEVICE)
        rating = batch["rating"].to(DEVICE)
        
        # Forward pass only
        pred = model(user, item)
        loss = criterion(pred, rating.float())
        
        test_loss += loss.item()
        num_batches += 1

avg_test_loss = test_loss / num_batches
print(f"\nTest Loss: {avg_test_loss:.4f}")