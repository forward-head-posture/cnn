# pylint: disable=no-member

import pytest
import cnn
from cnn.app import main, count_images


# @pytest.mark.skip
def test_main_with_one_step(mocker):
    mocker.patch("cnn.app.count_images").return_value = 1
    main(
        epochs=1,
        batch_size=1,
        # model_dir="s3://test/forward-head-posture/keras_ckpt",
    )


@pytest.mark.skip
def test_count_images():
    data_dir = "s3://tfrecord/forward-head-posture"
    train_counts = count_images(data_dir, "**/*train*")
    print(train_counts)
    validation_counts = count_images(data_dir, "**/*validation*")
    print(validation_counts)


# @pytest.mark.skip
def test_main_with_mock(mocker):
    mocker.patch("cnn.app.run_keras")
    num_train_images = 8800
    num_validation_images = 2222
    mocker.patch("cnn.app.count_images").side_effect = [
        num_train_images,
        num_validation_images,
    ]
    batch_size = 13
    main(batch_size=batch_size)
    kwargs = cnn.app.run_keras.call_args[1]

    assert kwargs["steps_per_epoch"] == num_train_images // batch_size
    assert kwargs["validation_steps"] == num_validation_images // batch_size
