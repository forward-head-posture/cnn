import os
import fire
import tensorflow as tf
from cnn.input_pipeline import input_fn
from cnn.model import get_model
from cnn.keras_callback import get_callbacks


def get_optimizer(name, learning_rate, decay_steps, decay_rate):
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        learning_rate, decay_steps=decay_steps, decay_rate=decay_rate
    )
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    if name == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)
    return tf.keras.optimizers.SGD(learning_rate=lr_schedule)


def compile_model(model, optimizer, loss_function="mean_squared_error"):
    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(),
            tf.keras.metrics.MeanSquaredError(),
        ],
    )


# pylint: disable=too-many-arguments
def run_keras(
    callbacks, epochs, get_input_fn, model, steps_per_epoch, validation_steps
):
    model.fit(
        get_input_fn(True)(),
        epochs=epochs,
        verbose=2,
        steps_per_epoch=steps_per_epoch,
        validation_data=get_input_fn(False)(),
        validation_steps=validation_steps,
        callbacks=callbacks,
    )


def count_images(data_dir, pattern="**/*train*"):
    glob_pattern = os.path.join(data_dir, pattern)
    return len(tf.io.gfile.glob(glob_pattern)) * 100


# pylint: disable=too-many-arguments
def main(
    batch_size=100,
    data_dir="s3://tfrecord/forward-head-posture",
    decay_rate=0.7,
    decay_steps=100,
    epochs=100,
    learning_rate=0.0001,
    model_dir=None,
    model_name="MobileNet",
    optimizer_name="adam",
):
    optimizer = get_optimizer(
        optimizer_name, learning_rate, decay_steps, decay_rate
    )
    model = get_model(model_name)
    compile_model(model, optimizer)
    callbacks = get_callbacks(model_dir)

    def get_input_fn(is_training):
        return lambda: input_fn(data_dir, batch_size, is_training)

    steps_per_epoch = count_images(data_dir, "**/*train*") // batch_size
    validation_steps = count_images(data_dir, "**/*validation*") // batch_size

    run_keras(
        callbacks=callbacks,
        epochs=epochs,
        get_input_fn=get_input_fn,
        model=model,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
    )


if __name__ == "__main__":
    fire.Fire(main)
