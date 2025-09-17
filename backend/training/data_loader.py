import numpy as np
import torch
from torch.utils.data import Dataset
import json

# Setup path to import from app module
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.utils import db

class RehabDataset(Dataset):
    """
    PyTorch Dataset to load rehab data from TimescaleDB.
    Can operate in two modes:
    - Unsupervised (supervised=False): Returns only feature windows.
    - Supervised (supervised=True): Returns feature windows and corresponding labels.
    """
    def __init__(self, window_size: int, step: int = 1, user_id: str = "ground_truth_trainer", supervised: bool = False):
        """
        Initializes the dataset.
        Args:
            window_size (int): The number of frames in each sample.
            step (int): The step size for the sliding window.
            user_id (str): The user ID to fetch data for.
            supervised (bool): Flag to determine if labels should be loaded.
        """
        self.window_size = window_size
        self.step = step
        self.user_id = user_id
        self.supervised = supervised

        print("Fetching data from database...")
        all_features, all_labels = self._fetch_data_from_db(self.user_id)

        if len(all_features) < self.window_size:
            raise ValueError(f"Not enough data ({len(all_features)} frames) to create a window of size {self.window_size}.")

        print("Creating sliding windows...")
        self.feature_windows = self._create_sliding_windows(all_features, self.window_size, self.step)

        if self.supervised:
            if not all_labels or all(label is None for label in all_labels):
                 raise ValueError("Supervised mode selected, but no labels found in the database. Please run the auto_labeler script first.")
            # We only need the label of the LAST frame in the window for prediction
            # Slicing the labels to match the number of windows created
            end_indices = range(self.window_size - 1, len(all_labels), self.step)
            self.labels = [all_labels[i] for i in end_indices if i < len(all_labels)]
            # Ensure the number of labels matches the number of windows
            self.labels = self.labels[:len(self.feature_windows)]

        print(f"Dataset created with {len(self.feature_windows)} samples.")

    def _fetch_data_from_db(self, user_id: str):
        """Fetches feature vectors and labels for a given user."""
        # The SQL query now correctly joins tables to filter by user_id
        sql = """
        SELECT f.feature_vector, f.labels
        FROM frames AS f
        JOIN sessions AS s ON f.session_id = s.session_id
        WHERE s.user_id = %s
        ORDER BY f.time; 
        """
        print(f"Executing query for user: {user_id}")
        results = []
        with db.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (user_id,))
                results = cur.fetchall()
        
        if not results:
            print("Warning: No data found in the database for the specified user.")
            return [], []

        feature_vectors = [row[0] for row in results]
        # Labels can be None if auto-labeler hasn't run, handle this case
        labels = [row[1] if row[1] is not None else {} for row in results]
        return np.array(feature_vectors, dtype=np.float32), labels

    def _create_sliding_windows(self, data: np.ndarray, window_size: int, step: int) -> np.ndarray:
        """Creates sliding windows from the time series data."""
        shape = (data.shape[0] - window_size + 1, window_size, data.shape[1])
        strides = (data.strides[0], data.strides[0], data.strides[1])
        windows = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
        return windows[::step]

    def __len__(self):
        return len(self.feature_windows)

    def __getitem__(self, idx: int):
        window_np = self.feature_windows[idx]
        
        if not self.supervised:
            return torch.from_numpy(window_np)
        else:
            label_data = self.labels[idx]
            # Extract severity: we take the first value found in the severity dict, or 0.0 if empty.
            severity_value = next(iter(label_data.get('severity', {}).values()), 0.0)
            
            labels = {
                "class": torch.tensor(label_data.get('class', 0), dtype=torch.long),
                "regression": torch.tensor([severity_value], dtype=torch.float32)
            }
            return torch.from_numpy(window_np), labels
