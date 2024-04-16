import pytest
from ryomen.main import Slicer, _nd_generator
import numpy as np


def test_non_array_input():
    with pytest.raises(ValueError):
        Slicer(image=[[1,2,3,4], [1,2,3,4, 5], [1, 2, 3, 4, 5]], crop_size=(2,), overlap=(0,0))



def test_total_3d_ind_coverage():
    image = np.zeros((100, 100, 10))
    for ind, xyz in _nd_generator(
        crop=[10, 10, 10], overlap=[2, 2, 2], shape=image.shape
    ):
        image[tuple(ind)] = 1
    assert np.all(image == 1)


def test_total_2d_ind_coverage():

    image = np.zeros((100, 100))
    for ind, xyz in _nd_generator(crop=[30, 30], overlap=[10, 10], shape=image.shape):
        image[tuple(ind)] = 1
    assert np.all(image == 1)


def test_total_5d_ind_coverage():

    image = np.zeros((30, 30, 30, 30, 30))
    for ind, xyz in _nd_generator(
        crop=[10, 10, 10, 10, 10], overlap=[2,] * 5, shape=image.shape
    ):
        image[tuple(ind)] = 1
    assert np.all(image == 1)

def test_source_destination():
    image = np.random.randn(1, 100, 100, 100)
    output = np.random.randn(1, 100, 100, 100)
    crop = (10, 10, 10)
    overlap = [2, 2, 2]

    crop_iterator = Slicer(image, crop_size=crop, overlap=overlap)
    for crop, source, destination in crop_iterator:
        a = output[destination]
        b = crop[source]
        assert a.shape == b.shape, source

def test_all_crops_same_size():
    image = np.random.randn(100, 100, 100)
    output = np.random.randn(100, 100, 100)
    cropsize = (10, 10, 10)
    overlap = [2, 2, 2]

    def same_size(t1, t2):
        return all(a == b for a, b in zip(t1, t2))

    crop_iterator = Slicer(image, crop_size=cropsize, overlap=overlap)
    crops = []
    for crop, source, destination in crop_iterator:
        crops.append(crop)
    assert all([same_size(c.shape, cropsize) for c in crops]), f'{source=}, {destination=}'


def test_nonfunctional_runtime():
    image = np.random.randn(1, 100, 100, 100)
    crop = (10, 10, 10)
    overlap = [2, 2, 2]

    crop_iterator = Slicer(image, crop_size=crop, overlap=overlap)
    for crop, source, destination in crop_iterator:
        pass


def test_2D_cropping():
    image = np.random.randn(100, 100)
    crop = (10, 10)
    overlap = [2, 2]

    crop_iterator = Slicer(image, crop_size=crop, overlap=overlap)
    for crop, source, destination in crop_iterator:
        pass


def test_3D_cropping():
    image = np.random.randn(100, 100, 10)
    crop = (10, 10, 10)
    overlap = [2, 2, 2]

    crop_iterator = Slicer(image, crop_size=crop, overlap=overlap)
    for crop, source, destination in crop_iterator:
        pass


def test_4D_cropping():
    image = np.random.randn(20, 20, 20, 20)
    crop = (10, 10, 10, 3)
    overlap = [2, 2, 2, 1]

    crop_iterator = Slicer(image, crop_size=crop, overlap=overlap)
    for crop, source, destination in crop_iterator:
        pass


def test_no_overlap():
    image = np.random.randn(1, 100, 100, 100)
    crop = (10, 10, 10)
    overlap = [0, 0, 0]

    crop_iterator = Slicer(image, crop_size=crop, overlap=overlap)
    for crop, source, destination in crop_iterator:
        pass


def test_strange_crop():
    image = np.random.randn(1, 10, 10, 10)
    output = np.empty_like(image)
    crop = (5, 5, 5)
    overlap = [0, 0, 0]

    crop_iterator = Slicer(image, crop_size=crop, overlap=overlap)
    for crop, source, destination in crop_iterator:
        output[destination] = crop[source]
    
    assert np.allclose(image, output)



def test_strangely_large_crop():
    image = np.random.randn(1, 100, 100, 100)
    crop = (1000, 1000, 1000)
    overlap = [10, 10, 10]

    crop_iterator = Slicer(image, crop_size=crop, overlap=overlap)
    for crop, source, destination in crop_iterator:
        assert np.allclose(image, crop)


def test_for_valid_inputs():
    image = np.random.randn(1, 100, 100, 100)
    crop = (1000, 1000, 1000)
    overlap = [10, 10, 10]
    with pytest.raises(ValueError):
        Slicer(image, (-1, -1, -1), (0, 0, 0))
    with pytest.raises(ValueError):
        Slicer(image, (-1, -1, -1), (342.9, 1000, -10))
    with pytest.raises(ValueError):
        Slicer(image, (10, 10, 10), (2, 2, 2), batch_size=0)
    with pytest.raises(ValueError):
        Slicer(image, (10, 10, 10), (2, 2, 2), batch_size=-1)
    with pytest.raises(RuntimeError):
        Slicer(image, (10, 10, 10), (2, 2, 2), batch_size=99999999999)
    with pytest.raises(TypeError):
        Slicer(image, (10, 10, 10), (2, 2, 2), batch_size=float("inf"))
    with pytest.raises(TypeError):
        Slicer(image, (10, 10, 10), (2, 2, 2), batch_size=float("nan"))
    with pytest.raises(TypeError):
        Slicer(image, (10, 10, 10), (2, 2, 2), batch_size=0.2)

def test_batch_size():
    image = np.random.randn(100, 100, 90)
    output = np.empty_like(image)
    crop = (10, 10, 10)
    overlap = [0, 0, 0]

    crop_iterator = Slicer(image, crop_size=crop, overlap=overlap, batch_size=10)
    for crop, source, destination in crop_iterator:
        assert len(crop) == len(source) == len(destination) == 10


def test_collate_fn():
    image = np.random.randn(100, 100, 90)
    output = np.empty_like(image)
    crop = (10, 10, 10)
    overlap = [0, 0, 0]

    collate = lambda x: np.stack(x, axis=0)

    crop_iterator = Slicer(image, crop_size=crop, overlap=overlap, collate=collate, batch_size=10)
    for crop, source, destination in crop_iterator:
        assert crop.shape[0] == 10


def test_batched_source_destination_agreement():
    image = np.random.randn(20, 10, 10)
    output = np.empty_like(image)
    crop = (10, 10, 10)
    overlap = [0, 0, 0]

    def fn(x):
        return x * 100 - 3

    crop_iterator = Slicer(image, crop_size=crop, overlap=overlap, output_transform=fn, batch_size=10)
    for crop, source, destination in crop_iterator:
        for _crop, _source, _destination in zip(crop, source, destination):
            output[_destination] = _crop[_source]
            assert np.allclose(_crop[_source], image[_destination] * 100 - 3)

    assert np.allclose((image * 100) - 3, output)


def test_src_dest_zero_overlap():
    image = np.random.randn(20, 100, 100)
    output = np.empty_like(image)
    crop = (10, 10, 10)
    overlap = [0, 0, 0]

    crop_iterator = Slicer(image, crop_size=crop, overlap=overlap)
    for crop, source, destination in crop_iterator:
        output[destination] = crop[source]

    assert np.allclose(image, output)

def test_source_destination_identity_agreement():
    import numpy as np

    image = np.random.randn(20, 10, 10)
    output = np.empty_like(image)
    crop = (10, 10, 10)
    overlap = [0, 0, 0]

    def fn(x):
        return x

    crop_iterator = Slicer(image, crop_size=crop, overlap=overlap, output_transform=fn)
    for crop, source, destination in crop_iterator:
        output[destination] = crop[source]
        assert np.allclose(crop[source], image[destination])

def test_output_fn():
    image = np.random.randn(20, 10, 10)
    output = np.empty_like(image)
    crop = (10, 10, 10)
    overlap = [0, 0, 0]

    def fn(x):
        return x * 100 - 3

    crop_iterator = Slicer(image, crop_size=crop, overlap=overlap, output_transform=fn)
    for crop, source, destination in crop_iterator:
        output[destination] = crop[source]
        assert np.allclose(crop[source], image[destination] * 100 - 3)

    assert np.allclose((image * 100) - 3, output)