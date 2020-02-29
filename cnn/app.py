import fire
import tensorflow as tf
from cnn.input_pipeline import input_fn
from cnn.model import get_model
from cnn.keras_callback import get_callbacks

# import os
# from cnn.metrics import get_metrics


def get_optimizer(name, learning_rate, decay_steps, decay_rate):
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        learning_rate, decay_steps=decay_steps, decay_rate=decay_rate
    )
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    if name == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)
    return tf.keras.optimizers.SGD(learning_rate=lr_schedule)


def _model(model_name, optimizer):
    model = get_model(model_name)
    model.compile(
        optimizer=optimizer,
        loss="mean_squared_error",
        # metrics=get_metrics(),
    )
    return model


def run_keras(model, model_dir, get_input_fn):
    # steps_per_epoch = NUM_TRAIN_IMAGE // FLAGS.batch_size + 1
    # validation_steps = NUM_EVAL_IMAGE // FLAGS.batch_size + 1
    epochs = 1
    steps_per_epoch = 1
    validation_steps = 1
    model.fit(
        get_input_fn(True)(),
        epochs=epochs,
        verbose=2,
        steps_per_epoch=steps_per_epoch,
        validation_data=get_input_fn(False)(),
        validation_steps=validation_steps,
        callbacks=get_callbacks(model_dir),
    )


# pylint: disable=too-many-arguments
def main(
    data_dir="s3://tfrecord/forward-head-posture",
    batch_size=32,
    learning_rate=0.0001,
    decay_rate=0.7,
    decay_steps=100,
    model_dir="s3://model-dir/forward-head-posture",
    model_name="InceptionV3",
    optimizer_name="rmsprop",
):
    optimizer = get_optimizer(
        optimizer_name, learning_rate, decay_steps, decay_rate
    )
    model = _model(model_name, optimizer)

    def get_input_fn(is_training):
        return lambda: input_fn(data_dir, batch_size, is_training)

    run_keras(model, model_dir, get_input_fn)


if __name__ == "__main__":
    fire.Fire(main)
