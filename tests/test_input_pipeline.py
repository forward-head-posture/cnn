from cnn.input_pipeline import input_fn


def test_input_pipeline():
    dataset = input_fn("s3://tfrecord/forward-head-posture/2020-02-15", 1)
    for image, distance in dataset.take(1).as_numpy_iterator():
        print(image)
        print(distance)
