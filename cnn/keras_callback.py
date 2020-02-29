# pylint: disable=dangerous-default-value,no-self-use

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
        print("epoch={}".format(epoch))
        self.print_log(logs)


def get_callbacks(model_dir):
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        model_dir,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
    )
    tb_cb = tf.keras.callbacks.TensorBoard(
        log_dir=model_dir, write_graph=False, update_freq=90
    )
    log_cb = Logging()
    stop_cb = tf.keras.callbacks.EarlyStopping()
    return [stop_cb, log_cb, tb_cb, ckpt_cb]
