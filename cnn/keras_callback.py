# pylint: disable=dangerous-default-value,no-self-use

import os
import logging
import tensorflow as tf

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


class Logging(tf.keras.callbacks.Callback):
    def print_log(self, logs):
        for k, v in logs.items():
            v = str(v).strip()
            logging.info(f"{k}={v}")

    def on_batch_end(self, batch, logs={}):
        if batch % 20 != 0:
            return
        self.print_log(logs)

    def on_epoch_end(self, epoch, logs={}):
        logging.info(f"epoch={epoch * -1}")
        self.print_log(logs)


def get_callbacks(model_dir):
    print(model_dir)
    log_cb = Logging()
    stop_cb = tf.keras.callbacks.EarlyStopping(baseline=500, patience=5)
    nan_cb = tf.keras.callbacks.TerminateOnNaN()
    callbacks = [log_cb, nan_cb, stop_cb]
    if not model_dir:
        return callbacks
    hostname = os.environ.get("HOSTNAME")
    log_dir = os.path.join(model_dir, hostname)
    tb_cb = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, write_graph=False, update_freq=90
    )
    ckptpath = os.path.join(log_dir, hostname)
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        ckptpath, monitor="val_loss", save_best_only=True
    )
    callbacks.extend([ckpt_cb, tb_cb])
    return callbacks
