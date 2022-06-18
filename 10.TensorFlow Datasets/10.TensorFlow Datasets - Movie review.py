import os
import matplotlib.pyplot

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds

(ds_train, ds_test), ds_info = tfds.load(
    "imdb_reviews",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

tokenize = tfds.features.text.Tokenizer()


def build_vocabulary():
    vocabulary = set()
    for text, _ in ds_train:
        vocabulary.update(tokenize.tokenize(text.numpy().lower()))
    return vocabulary


vocabulary = build_vocabulary()

encoder = tfds.features.text.TokenTextEncoder(
    vocabulary, oov_token="<UNK>", lowercase=True, tokenizer=tokenize
)


def my_encoding(text_tensor, label):
    return encoder.encode(text_tensor.numpy()), label


def encode_map(text, label):
    encoded_text, label = tf.py_function(
        my_encoding, inp=[text, label], Tout=(tf.int64, tf.int64)
    )

    encoded_text.set_shape([None])
    label.set_shape([])

    return encoded_text, label


AUTOTUNE = tf.data.experimental.AUTOTUNE
ds_train = ds_train.map(encode_map, num_parallel_calls=AUTOTUNE).cache()
ds_train = ds_train.shuffle(10000)
ds_train = ds_train.padded_batch(32, padded_shapes=([None], ()))
ds_train = ds_train.prefetch(AUTOTUNE)

ds_test = ds_test.map(encode_map)
ds_test = ds_test.padded_batch(32, padded_shapes=([None], ()))

model = keras.Sequential([
    layers.Masking(mask_value=0),
    layers.Embedding(input_dim=len(vocabulary) + 2, output_dim=32),
    # BATCH_SIZE * 1000 - > BATCH_SIZE x 1000 x 32
    layers.GlobalAveragePooling1D(),
    # BATCH_SIZE x 32
    layers.Dense(64, activation='relu'),
    layers.Dense(1),  # less than 0 negative, greater or equal to than positive
])

model.compile(
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(32 - 4, clipnorm=1),
    metrics=["accuracy"],
)

model.fit(ds_train, epochs=10, verbose=2)
model.evaluate(ds_test)
