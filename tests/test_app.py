import pytest
from cnn.app import main


@pytest.mark.skip
def test_main():
    main(batch_size=1, model_dir="s3://test/forward-head-posture/keras_ckpt")
