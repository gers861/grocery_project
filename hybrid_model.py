import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable
from sklearn.model_selection import train_test_split
from itertools import combinations
import pickle
from collections import defaultdict

# Set a random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- 1. Load and Preprocess Data ---
# Note: You need to have 'Groceries_cleaned_dataset2.csv' in the same directory
print("Loading and preprocessing data...")
# Correctly load the data and handle the date column
df = pd.read_csv("grocery_project\data\Groceries_cleaned_dataset2.csv")
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# Correctly rename columns for clarity, using the original column names
df = df.rename(columns={'Member_number': 'user', 'itemDescription': 'item'})

# Map users and items to unique integer IDs, ensuring they are strings for consistency
unique_users = df['user'].astype(str).unique()
unique_items = df['item'].astype(str).unique()

user2idx = {user: idx for idx, user in enumerate(unique_users)}
item2idx = {item: idx for idx, item in enumerate(unique_items)}
idx2item = {idx: item for item, idx in item2idx.items()}

df['user'] = df['user'].astype(str).map(user2idx)
df['item'] = df['item'].astype(str).map(item2idx)

num_users = len(unique_users)
num_items = len(unique_items)

print(f"Number of users: {num_users}")
print(f"Number of items: {num_items}")

# --- 2. Calculate Co-occurrence and Popularity (for fallback) ---
print("Calculating co-occurrence and popular items...")
baskets = df.groupby('user')['item'].apply(list).reset_index()

co_occurrence = defaultdict(lambda: defaultdict(int))
item_counts = defaultdict(int)

for items in baskets['item']:
    items = sorted(list(set(items)))
    for i, j in combinations(items, 2):
        co_occurrence[i][j] += 1
        co_occurrence[j][i] += 1
    for item in items:
        item_counts[item] += 1

co_occurrence_df = pd.DataFrame.from_dict(co_occurrence, orient='index').fillna(0).astype(int)
popular_items = pd.Series(item_counts).sort_values(ascending=False)

# --- 3. Create Training Pairs (with user) ---
print("Creating training pairs with user IDs...")
all_user_baskets = df.groupby(['user', 'Date'])['item'].apply(list)

positive_pairs = []
for (user_id, _), items in all_user_baskets.items():
    items = sorted(set(items))
    if len(items) > 1:
        positive_pairs.extend([(user_id, i, j) for i, j in combinations(items, 2)])

pos_samples = [(u, i, j, 1) for u, i, j in positive_pairs]
positive_set = set(positive_pairs)

neg_samples = []
while len(neg_samples) < len(pos_samples):
    u = np.random.randint(num_users)
    i = np.random.randint(num_items)
    j = np.random.randint(num_items)
    if i != j and (u, i, j) not in positive_set and (u, j, i) not in positive_set:
        neg_samples.append((u, i, j, 0))

all_samples = pos_samples + neg_samples
np.random.shuffle(all_samples)

df_samples = pd.DataFrame(all_samples, columns=['user', 'item1', 'item2', 'label'])

# --- 4. Train/test split and Dataset creation ---
print("Creating TensorFlow datasets...")
train_df, test_df = train_test_split(df_samples, test_size=0.2, random_state=42)

def df_to_dataset(df):
    return tf.data.Dataset.from_tensor_slices((
        {
            "user": df['user'].values,
            "item1": df['item1'].values,
            "item2": df['item2'].values
        },
        df['label'].values
    )).shuffle(10000).batch(256).prefetch(tf.data.AUTOTUNE)

train_ds = df_to_dataset(train_df)
test_ds = df_to_dataset(test_df)

# --- 5. Define Hybrid Model ---
# The model now accepts the number of users
@register_keras_serializable()
class HybridRecModel(tf.keras.Model):
    def __init__(self, num_users, num_items, embedding_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        
        # User and item embedding layers
        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_dim)
        self.item_embedding = tf.keras.layers.Embedding(num_items, embedding_dim)

    def call(self, inputs):
        # Retrieve embeddings for user, item1, and item2
        user_emb = self.user_embedding(inputs['user'])
        item1_emb = self.item_embedding(inputs['item1'])
        item2_emb = self.item_embedding(inputs['item2'])
        
        # Personalized dot product
        emb1 = item1_emb + user_emb
        emb2 = item2_emb + user_emb
        dot = tf.reduce_sum(emb1 * emb2, axis=1)
        
        return tf.nn.sigmoid(dot)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_users": self.num_users,
            "num_items": self.num_items,
            "embedding_dim": self.embedding_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# --- 6. Train model ---
print("Training the model...")
model = HybridRecModel(num_users=num_users, num_items=num_items, embedding_dim=64)

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=[tf.keras.metrics.BinaryAccuracy()]
)

history = model.fit(train_ds, validation_data=test_ds, epochs=10)

# --- 7. Save model & mappings ---
print("Saving model and mappings...")
model.save("hybrid_model.keras")

mappings = {
    'user2idx': user2idx,
    'item2idx': item2idx,
    'idx2item': idx2item,
    'co_occurrence_df': co_occurrence_df,
    'popular_items': popular_items
}

with open("mappings.pkl", "wb") as f:
    pickle.dump(mappings, f)

# --- 8. Hybrid Recommendation Function ---
def recommend_for_user(user_raw_id, user_basket_names, top_k=5, alpha=0.5):
    """
    Generates recommendations for a user based on a hybrid approach.
    Combines model-based similarity (learned embeddings) and
    co-occurrence/popularity heuristics.

    Args:
        user_raw_id (int): The raw user ID.
        user_basket_names (list): A list of item names in the user's current basket.
        top_k (int): Number of recommendations to return.
        alpha (float): Weighting factor for model-based score vs. heuristic score.
                       0 <= alpha <= 1. Higher alpha prioritizes the model.
    Returns:
        list: A list of recommended item names.
    """
    # Check if the user ID exists in the data, casting it to a string for the lookup
    if str(user_raw_id) not in user2idx:
        print(f"User {user_raw_id} not in data. Recommending popular items.")
        return [idx2item[i] for i in popular_items.index[:top_k]]

    user_idx = user2idx[str(user_raw_id)]
    
    # Filter out items not in the mapping, and handle type inconsistencies
    basket_idxs = []
    for item_name in user_basket_names:
        # Check if the item_name is a string before looking it up
        if isinstance(item_name, str) and item_name in item2idx:
            basket_idxs.append(item2idx[item_name])
        else:
            print(f"Warning: Item '{item_name}' not found or has an incorrect type. Skipping.")
    
    if not basket_idxs:
        print("Basket is empty or contains no recognized items. Recommending popular items.")
        return [idx2item[i] for i in popular_items.index[:top_k]]

    all_scores = {}
    
    # Exclude items already in the basket
    all_item_idxs = set(range(num_items))
    items_to_rank = list(all_item_idxs - set(basket_idxs))
    
    # --- Calculate Model-Based Scores ---
    model_scores = defaultdict(list)
    user_tensor_base = tf.constant([user_idx], dtype=tf.int32)
    item1_tensor = tf.constant(items_to_rank, dtype=tf.int32)
    
    for basket_item_idx in basket_idxs:
        user_tensor = tf.tile(user_tensor_base, [len(items_to_rank)])
        item2_tensor = tf.constant([basket_item_idx] * len(items_to_rank), dtype=tf.int32)
        
        inputs = {"user": user_tensor, "item1": item1_tensor, "item2": item2_tensor}
        preds = model(inputs).numpy()
        for i, score in zip(items_to_rank, preds):
            model_scores[i].append(score)
            
    avg_model_scores = {item: np.mean(scores) for item, scores in model_scores.items()}

    # --- Calculate Heuristic Scores ---
    heuristic_scores = {}
    pop_max = popular_items.max()
    co_occurrence_max = co_occurrence_df.values.max()

    for candidate_idx in items_to_rank:
        co_score = 0
        for basket_item_idx in basket_idxs:
            if basket_item_idx in co_occurrence_df.columns and candidate_idx in co_occurrence_df.index:
                co_score += co_occurrence_df.loc[candidate_idx, basket_item_idx]
        
        normalized_co_score = co_score / (co_occurrence_max + 1e-6)
        normalized_pop_score = popular_items.get(candidate_idx, 0) / (pop_max + 1e-6)
        
        heuristic_scores[candidate_idx] = normalized_co_score + normalized_pop_score

    # --- Combine scores ---
    final_scores = {}
    for item_idx in items_to_rank:
        model_s = avg_model_scores.get(item_idx, 0)
        heuristic_s = heuristic_scores.get(item_idx, 0)
        final_scores[item_idx] = alpha * model_s + (1 - alpha) * heuristic_s

    sorted_items = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    top_indices = [item[0] for item in sorted_items[:top_k]]

    return [idx2item[i] for i in top_indices]

# --- 9. Example usage ---
print("\n--- Generating sample recommendation ---")
user_id = 1808
basket = ["whole milk", "rolls/buns"]
recommendations = recommend_for_user(user_id, basket, top_k=5)
print(f"Recommendations for User {user_id} with basket {basket}: {recommendations}")
