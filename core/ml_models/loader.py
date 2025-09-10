# core/ml_models/loader.py
import os
import pickle
import numpy as np
import tensorflow as tf
from .hybrid_model import HybridRecModel

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "hybrid_model.keras")
MAPPINGS_PATH = os.path.join(BASE_DIR, "mappings.pkl")

model = None
user2idx = None
item2idx = None
idx2item = None
num_users = None
num_items = None
item2idx_norm = None  # normalized-name -> idx
item_emb_matrix = None  # cached item embeddings (for cold-start item-item)

def _normalize_item(s: str) -> str:
    return (s or "").strip().lower()

def load_resources():
    global model, user2idx, item2idx, idx2item, num_users, num_items, item2idx_norm, item_emb_matrix

    if model is not None:
        return model, user2idx, item2idx, idx2item

    # Load mappings (3-tuple or 5-tuple—support both)
    with open(MAPPINGS_PATH, "rb") as f:
        loaded = pickle.load(f)
        if isinstance(loaded, tuple) and len(loaded) >= 3:
            user2idx, item2idx, idx2item = loaded[:3]
        else:
            raise ValueError("mappings.pkl has unexpected format")

    num_users = len(user2idx)
    num_items = len(item2idx)

    # Build normalized lookup so any casing/spacing works
    item2idx_norm = {_normalize_item(name): idx for name, idx in item2idx.items()}

    # Load model
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"HybridRecModel": HybridRecModel},
    )

    # Cache item embeddings (for item–item fallback)
    # shape: (num_items, embedding_dim)
    item_emb_matrix = model.item_embedding(np.arange(num_items)).numpy()
    # L2-normalize for cosine similarity
    norms = np.linalg.norm(item_emb_matrix, axis=1, keepdims=True) + 1e-8
    item_emb_matrix = item_emb_matrix / norms

    return model, user2idx, item2idx, idx2item

def _to_item_idx(name: str):
    """Return index for normalized item name, or None."""
    if name is None:
        return None
    k = _normalize_item(name)
    return item2idx_norm.get(k)

def _item_item_fallback(basket_idxs, top_k=5):
    """
    Cold-start / unknown user recommendation using item–item cosine similarity
    from learned item embeddings.
    """
    if not basket_idxs:
        # Nothing known in basket — as a last resort, return the most “central” items by avg similarity
        sims = item_emb_matrix @ item_emb_matrix[basket_idxs].T if basket_idxs else None
        # If we truly have nothing, default to first K items deterministically
        candidate_idxs = list(range(num_items))[:top_k]
        return [idx2item[i] for i in candidate_idxs]

    # Average cosine similarity to basket items
    basket_vec = item_emb_matrix[basket_idxs]  # (B, d)
    # cosine sim for all items vs basket items: (N, d) @ (d, B) -> (N, B)
    sims = item_emb_matrix @ basket_vec.T
    mean_sims = sims.mean(axis=1)  # (N,)

    # Exclude items already in basket
    mean_sims[basket_idxs] = -1e9

    # Top-k by similarity, deterministic tie-break by index
    top = np.argsort(np.vstack([-mean_sims, np.arange(num_items)]).T, axis=0)  # not great
    # Better deterministic sort:
    candidates = sorted(range(num_items), key=lambda i: (-mean_sims[i], i))
    top_idx = candidates[:top_k]
    return [idx2item[i] for i in top_idx]

def recommend_for_user(user_raw_id, basket_names, top_k=5):
    """
    Unified recommend function that handles:
      - known user + known basket -> hybrid model score
      - known user + partially unknown basket -> use known subset; if none known -> item–item fallback
      - new user  -> item–item fallback
    """
    global model, user2idx, item2idx, idx2item, num_items
    if model is None:
        load_resources()

    # normalize + map basket to indices
    basket_idxs = []
    for n in basket_names or []:
        idx = _to_item_idx(n)
        if idx is not None:
            basket_idxs.append(idx)
    basket_idxs = sorted(set(basket_idxs))

    # known vs new user
    is_known_user = False
    try:
        user_id = int(user_raw_id)
        is_known_user = user_id in user2idx
    except Exception:
        is_known_user = False

    if not basket_idxs:
        # no known items → item–item fallback 
        return _item_item_fallback([], top_k=top_k)

    if not is_known_user:
        # new user → item–item fallback from basket
        return _item_item_fallback(basket_idxs, top_k=top_k)

    # Known user + known basket → use the hybrid model
    user_idx = user2idx[user_id]

    # Vectorized scoring
    scores = np.full(num_items, -1e9, dtype=np.float32)  # start very low to exclude by default
    basket_arr = np.array(basket_idxs, dtype=np.int32)
    user_arr = np.full_like(basket_arr, user_idx, dtype=np.int32)

    for item_idx in range(num_items):
        if item_idx in basket_idxs:
            continue  # never recommend what’s already in the basket
        item_arr = np.full_like(basket_arr, item_idx, dtype=np.int32)
        inputs = {"user": user_arr, "item1": item_arr, "item2": basket_arr}
        preds = model(inputs).numpy()
        scores[item_idx] = float(preds.mean())

    # Top-k by score, deterministic tie-break on index
    candidates = sorted(range(num_items), key=lambda i: (-scores[i], i))
    top_idx = candidates[:top_k]
    return [idx2item[i] for i in top_idx]

