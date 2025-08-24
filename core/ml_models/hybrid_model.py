import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
class HybridRecModel(tf.keras.Model):
    def __init__(self, num_users, num_items, embedding_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_dim)
        self.item_embedding = tf.keras.layers.Embedding(num_items, embedding_dim)

    def call(self, inputs):
        user_emb = self.user_embedding(inputs['user'])
        item1_emb = self.item_embedding(inputs['item1'])
        item2_emb = self.item_embedding(inputs['item2'])
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
