from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

from app.feature_adapters.common import backend_path
from app.models.base import BasePredictor
from app.services.errors import BackendError


@tf.keras.utils.register_keras_serializable()
class ProbSparseAttention(layers.Layer):
    def __init__(self, num_heads, key_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout = dropout
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)

    def call(self, query, key, value):
        u = tf.cast(tf.sqrt(tf.cast(tf.shape(query)[1], tf.float32)), tf.int32)
        idx = tf.random.shuffle(tf.range(tf.shape(query)[1]), seed=42)[:u]
        key_sample = tf.gather(key, idx, axis=1)
        value_sample = tf.gather(value, idx, axis=1)
        return self.attn(query, key_sample, value_sample)

    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads, "key_dim": self.key_dim, "dropout": self.dropout})
        return config


@tf.keras.utils.register_keras_serializable()
class EncoderLayer(layers.Layer):
    def __init__(self, embed_dim, num_heads, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.attn = ProbSparseAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(embed_dim * 4, activation="gelu"), layers.Dense(embed_dim)]
        )
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.dropout = layers.Dropout(dropout)

    def call(self, x):
        attn_out = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads, "dropout": self.dropout_rate})
        return config


@tf.keras.utils.register_keras_serializable()
class InformerEncoder(layers.Layer):
    def __init__(self, depth, embed_dim, num_heads, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.depth = depth
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.layers_ = [EncoderLayer(embed_dim, num_heads, dropout) for _ in range(depth)]
        self.pool = layers.AveragePooling1D(pool_size=2, strides=2, padding="same")

    def call(self, x):
        for layer in self.layers_:
            x = layer(x)
            x = self.pool(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "depth": self.depth,
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "dropout": self.dropout_rate,
            }
        )
        return config


class InformerPredictor(BasePredictor):
    def __init__(self, model_dir: str | Path):
        self.model_dir = backend_path(model_dir)
        model_path = self.model_dir / "best_drypoint_model.h5"
        if not model_path.exists():
            raise BackendError("MODEL_NOT_LOADED", f"missing model file: {model_path}", status_code=500)
        custom_objects = {
            "ProbSparseAttention": ProbSparseAttention,
            "EncoderLayer": EncoderLayer,
            "InformerEncoder": InformerEncoder,
        }
        self.model = load_model(model_path, custom_objects=custom_objects)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6), loss="mse")
        self.y_scaler = joblib.load(self.model_dir / "y_scaler.pkl")

    def predict(self, X):
        try:
            pred_scaled = self.model.predict(X, verbose=0).reshape(-1)[0]
            return float(self.y_scaler.inverse_transform(np.array([[pred_scaled]])).reshape(-1)[0])
        except Exception as exc:
            raise BackendError("PREDICTION_FAILED", str(exc), status_code=500) from exc
