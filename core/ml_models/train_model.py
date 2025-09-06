import os
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from itertools import combinations
from hybrid_model import HybridRecModel  # import your custom model

# ---------------------------
# 1. Load and preprocess data
# ---------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "../../data/Groceries_cleaned_dataset2.csv")

df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# Encode users/items
user_ids = df['Member_number'].unique()
item_names = df['itemDescription'].unique()

user2idx = {u: i for i, u in enumerate(user_ids)}
item2idx = {name: j for j, name in enumerate(item_names)}
idx2item = {v: k for k, v in item2idx.items()}

df['user'] = df['Member_number'].map(user2idx)
df['item'] = df['itemDescription'].map(item2idx)

num_users = len(user2idx)
num_items = len(item2idx)

# ---------------------------
# 2. Build training pairs
# ---------------------------
baskets = df.groupby(['user', 'Date'])['item'].apply(list).reset_index()

positive_pairs = []
for row in baskets.itertuples():
    items = sorted(set(row.item))
    if len(items) > 1:
        positive_pairs.extend([(row.user, i, j) for i, j in combinations(items, 2)])

pos_samples = [(u, i, j, 1) for u, i, j in positive_pairs]

positive_set = set((u, i, j) for u, i, j, _ in pos_samples)
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

# ---------------------------
# 3. Train/test split
# ---------------------------
train_df, test_df = train_test_split(df_samples, test_size=0.2, random_state=42)

def df_to_dataset(df):
    return tf.data.Dataset.from_tensor_slices((
        {"user": df['user'].values,
         "item1": df['item1'].values,
         "item2": df['item2'].values},
        df['label'].values
    )).shuffle(10000).batch(256).prefetch(tf.data.AUTOTUNE)

train_ds = df_to_dataset(train_df)
test_ds = df_to_dataset(test_df)

# ---------------------------
# 4. Train model
# ---------------------------
model = HybridRecModel(num_users=num_users, num_items=num_items)

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=[tf.keras.metrics.BinaryAccuracy()]
)

history = model.fit(train_ds, validation_data=test_ds, epochs=10)

# ---------------------------
# 5. Save model + mappings
# ---------------------------
MODEL_PATH = "hybrid_model.keras"
MAPPINGS_PATH = "mappings.pkl"

model.save(MODEL_PATH)   # Portable .keras format

# --- Cold-start helpers ---
from collections import Counter, defaultdict

co_occurrence = defaultdict(Counter)
for _, group in df.groupby("Member_number"):
    items_bought = group["itemDescription"].tolist()
    for i in range(len(items_bought)):
        for j in range(i + 1, len(items_bought)):
            item_a, item_b = items_bought[i], items_bought[j]
            co_occurrence[item_a][item_b] += 1
            co_occurrence[item_b][item_a] += 1

# Top-50 popular items for brand new users
popular_items = [item for item, _ in Counter(df["itemDescription"]).most_common(50)]

# Save all mappings + cold start helpers
with open(MAPPINGS_PATH, "wb") as f:
    pickle.dump((user2idx, item2idx, idx2item, co_occurrence, popular_items), f)

print(f"✅ Model saved to {MODEL_PATH}")
print(f"✅ Mappings saved to {MAPPINGS_PATH} with cold-start helpers")

