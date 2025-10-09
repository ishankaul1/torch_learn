"""
Goal here is to get a proper collab filtering model runnning on MovieLens 100k.
Mostly looking for comfort with pytorch syntax, workflow, tools etc.
"""

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split




OUT_DIR = "collab_filter/out/experiment_small"
RATINGS_CSV_PATH = "collab_filter/data/ml-latest-small/ratings.csv"

DEVICE = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


MODEL_OUTPUT_FILENAME = "collab_filter_model.pth"
METRICS_OUTPUT_FILENAME = "collab_filter_metrics.json"


# TODO - hyperparams

TRAIN_BATCH_SIZE = 256
EPOCHS = 50
MODEL_DIMS = 32


class RatingsDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        # Numpy arrs
        self.users = df["userId"].values
        self.items = df["movieId"].values
        self.ratings = df["rating"].values

    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return {
            "user": self.users[idx],
            "item": self.items[idx],
            "rating": self.ratings[idx],
        }

df = pd.read_csv(RATINGS_CSV_PATH)


user_to_idx = {user_id: idx for idx, user_id in enumerate(df["userId"].unique())}
item_to_idx = {item_id: idx for idx, item_id in enumerate(df["movieId"].unique())}



n_users = len(user_to_idx)
n_items = len(item_to_idx)



train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = RatingsDataset(train_df)
test_dataset = RatingsDataset(test_df)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=TRAIN_BATCH_SIZE)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=512)