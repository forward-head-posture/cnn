# pylint: disable=dangerous-default-value,no-self-use

import os
import tensorflow as tf


class Logging(tf.keras.callbacks.Callback):
    def print_log(self, logs):
        for k, v in logs.items():
            v = str(v).strip()
            print("{}={}".format(k, v))

    def on_batch_end(self, batch, logs={}):
        if batch % 20 != 0:
            return
        self.print_log(logs)

    def on_epoch_end(self, epoch, logs={}):
        print("epoch={}".format(epoch * -1))
        self.print_log(logs)


def get_callbacks(model_dir):
    log_cb = Logging()
    stop_cb = tf.keras.callbacks.EarlyStopping(min_delta=5, baseline=500)
    nan_cb = tf.keras.callbacks.TerminateOnNaN()
    callbacks = [log_cb, stop_cb, nan_cb]
    if not model_dir:
        return callbacks
    hostname = os.environ.get("HOSTNAME")
    log_dir = os.path.join(model_dir, hostname)
    tb_cb = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, write_graph=False, update_freq=90
    )
    ckptpath = os.path.join(log_dir, hostname)
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        ckptpath,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
    )
    callbacks.extend([tb_cb, ckpt_cb])
    return callbacks
