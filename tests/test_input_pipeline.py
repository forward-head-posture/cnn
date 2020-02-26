from cnn.input_pipeline import input_fn


def test_input_pipeline():
    dataset = input_fn("s3://tfrecord/forward-head-posture/2020-02-15", 1)
    t = dataset.take(1)
    print(dataset)
    for image, distance in t.as_numpy_iterator():
        print(image)
        print(distance[0])
