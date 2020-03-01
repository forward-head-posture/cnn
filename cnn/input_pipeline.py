import os
import tensorflow as tf
from cnn.inception_preprocessing import preprocess_image


def get_features(serialized):
    keys_to_features = {
        "image/encoded": tf.io.FixedLenFeature((), tf.string),
        "image/distance": tf.io.FixedLenFeature([], dtype=tf.float32),
    }
    features = tf.io.parse_single_example(serialized, keys_to_features)
    return features


def input_fn(data_dir, batch_size, is_training=True, crop=True):
    file_pattern = os.path.join(
        data_dir, "**/train-*" if is_training else "**/validation-*"
    )
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=is_training)

    dataset = dataset.repeat(None if is_training else 1)

    def fetch_dataset(filename):
        buffer_size = 8 * 1024 * 1024  # 8 MiB per file
        dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
        return dataset

    dataset = dataset.interleave(fetch_dataset, cycle_length=16)

    if is_training:
        dataset = dataset.shuffle(1024)

    def dataset_parser(serialized):
        features = get_features(serialized)
        image = tf.image.decode_jpeg(features["image/encoded"], 3)
        image = preprocess_image(image, 299, 299, is_training, crop_image=crop)
        distance = features["image/distance"] * 1000
        return image, distance

    dataset = dataset.map(dataset_parser, num_parallel_calls=2)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
