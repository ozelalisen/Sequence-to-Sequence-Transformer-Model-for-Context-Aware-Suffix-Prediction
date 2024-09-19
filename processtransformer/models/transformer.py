import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# import tensorflow_models as tfm


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layer_norm_a = layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_b = layers.LayerNormalization(epsilon=1e-6)
        self.dropout_a = layers.Dropout(rate)
        self.dropout_b = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout_a(attn_output, training=training)
        out_a = self.layer_norm_a(inputs + attn_output)
        ffn_output = self.ffn(out_a)
        ffn_output = self.dropout_b(ffn_output, training=training)
        return self.layer_norm_b(out_a + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, max_len, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=max_len, output_dim=embed_dim)

    def call(self, x):
        max_len = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=max_len, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


def positional_encoding(length, depth):
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10000**depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x, key=context, value=context, return_attention_scores=True
        )

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x


class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(query=x, value=x, key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(query=x, value=x, key=x, use_causal_mask=True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dff, activation="relu"),
                tf.keras.layers.Dense(d_model),
                tf.keras.layers.Dropout(dropout_rate),
            ]
        )
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout_rate
        )

        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self,
        *,
        max_len,
        num_layers,
        d_model,
        num_heads,
        dff,
        vocab_size,
        context_vocab_sizes,
        dropout_rate=0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = TokenAndPositionEmbedding(
            max_len=max_len, vocab_size=vocab_size, embed_dim=d_model
        )

        self.context_pos_embeddings = [
            TokenAndPositionEmbedding(
                max_len=max_len, vocab_size=vocab_size, embed_dim=d_model
            )
            for vocab_size in context_vocab_sizes
        ]

        self.enc_layers = [
            EncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dff=dff,
                dropout_rate=dropout_rate,
            )
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, x_c=None):
        x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.
        if x_c:
            x_c = [self.context_pos_embeddings[i](x) for i, x in enumerate(x_c)]
            x = layers.Concatenate(axis=1)([x] + x_c)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x  # Shape `(batch_size, seq_len, d_model)`.


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout_rate
        )

        self.cross_attention = CrossAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout_rate
        )

        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(
        self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1
    ):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(
                d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate
            )
            for _ in range(num_layers)
        ]

        self.last_attn_scores = None

    def call(self, x, context):
        # `x` is token-IDs shape (batch, target_seq_len)
        x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        # The shape of x is (batch_size, target_seq_len, d_model).
        return x


class Transformer(tf.keras.Model):
    def __init__(
        self,
        *,
        max_len,
        input_vocab_size,
        target_vocab_size,
        context_vocab_sizes,
        num_layers=2,
        d_model=36,
        num_heads=4,
        dff=64,
        dropout_rate=0.1
    ):
        super().__init__()
        self.encoder = Encoder(
            max_len=max_len,
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            vocab_size=input_vocab_size,
            context_vocab_sizes=context_vocab_sizes,
            dropout_rate=dropout_rate,
        )

        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            vocab_size=target_vocab_size,
            dropout_rate=dropout_rate,
        )

        self.suffix_layer = tf.keras.layers.Dense(
            target_vocab_size, name="suffix_activity", activation="softmax"
        )

        self.file_size = tf.keras.layers.Dense(
            1,
            name="suffix_file_size",
        )

    def call(self, inputs):
        context, x, x_c = inputs

        context = self.encoder(context, x_c)  # (batch_size, context_len, d_model)

        x = self.decoder(x, context)  # (batch_size, target_len, d_model)

        # Final linear layer output.
        logits_suffix = self.suffix_layer(
            x
        )  # (batch_size, target_len, target_vocab_size)

        logits_file_size = self.file_size(x)

        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            # b/250038731
            del logits_suffix._keras_mask
            del logits_file_size._keras_mask
        except AttributeError:
            pass

        # Return the final output and the attention weights.
        return {
            "suffix_activity": logits_suffix,
            "suffix_file_size": logits_file_size,
            "attn_scores": self.decoder.last_attn_scores,
        }


# def get_next_activity_model(
#     max_case_length,
#     context_size,
#     vocab_size,
#     context_vocab_size,
#     output_dim,
#     embed_dim=36,
#     num_heads=4,
#     ff_dim=64,
# ):
#     inputs = layers.Input(shape=(max_case_length,))
#     decoder_inputs = layers.Input(shape=(max_case_length,))
#     context_inputs = [layers.Input(shape=(max_case_length,))]
#
#     # x = tfm.nlp.models.TransformerEncoder(
#     #     num_layers=4,
#     #     num_attention_heads=num_heads,
#     #     intermediate_size=embed_dim * embed_multiplier,
#     #     dropout_rate=0.1,
#     #     attention_dropout_rate=0.1,
#     # )(x)
#     # x = tfm.nlp.models.TransformerDecoder(
#     #     num_layers=4,
#     #     num_attention_heads=num_heads,
#     #     intermediate_size=embed_dim * embed_multiplier,
#     #     dropout_rate=0.1,
#     # )(target=decoder_x, memory=x)
#     # outputs = Transformer(
#     #     num_layers=2,
#     #     d_model=64,
#     #     dff=128,
#     #     num_heads=4,
#     #     input_vocab_size=vocab_size,
#     #     target_vocab_size=output_dim,
#     #     maximum_position_encoding=max_case_length,
#     # )(inputs, decoder_inputs)
#
#     suffix, file_size = Transformer(
#         num_layers=4,
#         d_model=embed_dim,
#         num_heads=num_heads,
#         dff=ff_dim,
#         input_vocab_size=vocab_size,
#         target_vocab_size=output_dim,
#         dropout_rate=0.1,
#     )((inputs, decoder_inputs, context_inputs[0]))
#
#     # outputs = layers.Dense(output_dim)(x)
#     # outputs = tf.nn.softmax(outputs, axis=2)
#     transformer = tf.keras.Model(
#         inputs=[inputs, decoder_inputs] + context_inputs,
#         outputs=[suffix, file_size],
#         name="next_activity_transformer",
#     )
#
#     return transformer


def get_next_time_model(
    max_case_length, vocab_size, output_dim=1, embed_dim=36, num_heads=4, ff_dim=64
):

    inputs = layers.Input(shape=(max_case_length,))
    # Three time-based features
    time_inputs = layers.Input(shape=(3,))
    x = TokenAndPositionEmbedding(max_case_length, vocab_size, embed_dim)(inputs)
    x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x_t = layers.Dense(32, activation="relu")(time_inputs)
    x = layers.Concatenate()([x, x_t])
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(output_dim, activation="linear")(x)
    transformer = tf.keras.Model(
        inputs=[inputs, time_inputs], outputs=outputs, name="next_time_transformer"
    )
    return transformer


def get_remaining_time_model(
    max_case_length, vocab_size, output_dim=1, embed_dim=36, num_heads=4, ff_dim=64
):

    inputs = layers.Input(shape=(max_case_length,))
    # Three time-based features
    time_inputs = layers.Input(shape=(3,))
    x = TokenAndPositionEmbedding(max_case_length, vocab_size, embed_dim)(inputs)
    x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x_t = layers.Dense(32, activation="relu")(time_inputs)
    x = layers.Concatenate()([x, x_t])
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(output_dim, activation="linear")(x)
    transformer = tf.keras.Model(
        inputs=[inputs, time_inputs], outputs=outputs, name="remaining_time_transformer"
    )
    return transformer
