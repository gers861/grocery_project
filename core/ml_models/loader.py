import os
import pickle
import tensorflow as tf
from .hybrid_model import HybridRecModel
import numpy as np

# Paths relative to this file
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "hybrid_model.keras")
MAPPINGS_PATH = os.path.join(BASE_DIR, "mappings.pkl")

# Global variables to hold loaded objects
model = None
user2idx = None
item2idx = None
idx2item = None
num_users = None
num_items = None


def load_resources():
    """Load model and mappings into memory (only once)."""
    global model, user2idx, item2idx, idx2item, num_users, num_items

    if model is None:
        # Load mappings
        with open(MAPPINGS_PATH, "rb") as f:
            user2idx, item2idx, idx2item = pickle.load(f)

        num_users = len(user2idx)
        num_items = len(item2idx)

        # Load model with custom class
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={"HybridRecModel": HybridRecModel}
        )

    return model, user2idx, item2idx, idx2item


def recommend_for_user(user_raw_id, user_basket_names, top_k=5):
    """Generate top-k recommendations for a given user and basket."""
    global model, user2idx, item2idx, idx2item, num_items

    if model is None:
        load_resources()

    if user_raw_id not in user2idx:
        return []

    user_idx = user2idx[user_raw_id]
    basket_idxs = [item2idx[i] for i in user_basket_names if i in item2idx]

    if not basket_idxs:
        return []

    scores = []
    for item_idx in range(num_items):
        if item_idx in basket_idxs:
            continue

        inputs = {
    "user": np.array([user_idx] * len(basket_idxs)),
    "item1": np.array([item_idx] * len(basket_idxs)),
    "item2": np.array(basket_idxs)
}
        preds = model(inputs).numpy().mean()
        scores.append((item_idx, preds))

    # Sort by score
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

    return [idx2item[i] for i, _ in scores]
