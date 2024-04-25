import pytest
from ryomen.main import Slicer, _nd_generator
import numpy as np
import torch
import zarr
import itertools


@pytest.mark.parametrize('dtype,device,modifiers', itertools.product(
    [torch.uint8, torch.float16, torch.float32, torch.int32, torch.int16, torch.int8],
    ['cpu'],
    [lambda x: x.mul_(float('inf')), lambda x: x.__setitem__(x.gt(0.5), float('NaN'))]

))
def test_fuzz_pytorch_inputs(dtype, device, modifiers):
    image = torch.randn((6, 6)).to(dtype).to(device)
    try:
        modifiers(image)
    except:
        pass
    output = torch.empty_like(image)
    crop = (5, 5)
    overlap = (2, 2)

    crop_iterator = Slicer(
        image, crop_size=crop, overlap=overlap, batch_size=1, pad=True,
    )
    for crop, source, destination in crop_iterator:
        output[destination] = crop[source]

    if torch.any(torch.isnan(image)):
        ind0 = torch.isnan(image)
        ind1 = torch.isnan(output)
        assert torch.allclose(
            ind0, ind1
        )
        image[ind0] = 1
        output[ind1] = 1
        assert torch.allclose(image, output), f'\n{image}\n\n{output}\n\n{image-output}'
    else:
        assert torch.allclose(image, output), f'\n\n{image-output}'


@pytest.mark.parametrize('N,leading_dims', itertools.product([1, 2, 3, 4], [0, 1, 2, 3]))
def test_all_dim_numpy(N: int, leading_dims: int):
    s = 7
    image = np.arange(s ** N).reshape(([1] * leading_dims)+ ([s,] * N))
    print(image.shape)
    output = np.empty_like(image)
    crop = (5, ) * N
    overlap = (2, ) * N

    crop_iterator = Slicer(
        image, crop_size=crop, overlap=overlap, batch_size=1, pad=True,
    )
    for crop, source, destination in crop_iterator:
        output[destination] = crop[source]

    assert np.allclose(image[:], output), f'\n\n{image-output}'

@pytest.mark.parametrize('N,leading_dims', itertools.product([1, 2, 3, 4], [0, 1, 2, 3]))
def test_all_dim_torch(N: int, leading_dims: int):
    s = 7
    image = torch.arange(s ** N).reshape(([1] * leading_dims)+ ([s,] * N))
    output = torch.empty_like(image)
    crop = (5, ) * N
    overlap = (2, ) * N

    crop_iterator = Slicer(
        image, crop_size=crop, overlap=overlap, batch_size=1, pad=True,
    )
    for crop, source, destination in crop_iterator:
        output[destination] = crop[source]

    assert torch.allclose(image[:], output), f'\n\n{image-output}'

@pytest.mark.parametrize('N,leading_dims', itertools.product([1, 2, 3, 4], [0, 1, 2, 3]))
def test_all_dim_zarr(N: int, leading_dims: int):
    s = 7

    _image = np.arange(s ** N).reshape(([1] * leading_dims) + ([s, ] * N))
    image = zarr.zeros(([1] * leading_dims) + ([s, ] * N))
    image[:] = _image
    output = zarr.zeros(shape=([1] * leading_dims) + ([s, ] * N))
    crop = (5, ) * N
    overlap = (2, ) * N

    crop_iterator = Slicer(
        image, crop_size=crop, overlap=overlap, batch_size=1, pad=True,
    )
    for crop, source, destination in crop_iterator:
        output[destination] = crop[source]

    assert np.allclose(image[:], output[:]), f'\n\n{image-output}'


def test_that_nd_gen_finishes():
    max = 10000
    counter = 0
    for ind, xyz in _nd_generator(
        crop=[10, 10, 10], overlap=[2, 2, 2], shape=(100, 100, 10), pad=False
    ):
        counter += 1
        assert counter < max

    # now do it with padding
    max = 10000
    counter = 0
    for ind, xyz in _nd_generator(
        crop=[10, 10, 10], overlap=[2, 2, 2], shape=(100, 100, 10), pad=True
    ):
        counter += 1
        assert counter < max


def test_non_array_input():
    with pytest.raises(ValueError):
        Slicer(
            image=[[1, 2, 3, 4], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
            crop_size=(2,),
            overlap=(0, 0),
        )


def test_total_3d_ind_coverage():
    image = np.zeros((100, 100, 10))
    for ind, xyz in _nd_generator(
        crop=[10, 10, 10], overlap=[2, 2, 2], shape=image.shape, pad=False
    ):
        image[tuple(ind)] = 1
    assert np.all(image == 1)


def test_total_2d_ind_coverage():

    image = np.zeros((100, 100))
    for ind, xyz in _nd_generator(
        crop=[30, 30], overlap=[10, 10], shape=image.shape, pad=False
    ):
        image[tuple(ind)] = 1
    assert np.all(image == 1)


def test_total_5d_ind_coverage():

    image = np.zeros((30, 30, 30, 30, 30))
    for ind, xyz in _nd_generator(
        crop=[10, 10, 10, 10, 10], overlap=[2,] * 5, shape=image.shape, pad=False
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
    assert all(
        [same_size(c.shape, cropsize) for c in crops]
    ), f"{source=}, {destination=}"


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

    crop_iterator = Slicer(image, crop_size=crop, overlap=overlap, pad=False)
    for crop, source, destination in crop_iterator:
        assert np.allclose(image, crop), f'{crop.shape}'


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

    crop_iterator = Slicer(
        image, crop_size=crop, overlap=overlap, collate=collate, batch_size=10
    )
    for crop, source, destination in crop_iterator:
        assert crop.shape[0] == 10


def test_batched_source_destination_agreement():
    image = np.random.randn(20, 10, 10)
    output = np.empty_like(image)
    crop = (10, 10, 10)
    overlap = [0, 0, 0]

    def fn(x):
        return x * 100 - 3

    crop_iterator = Slicer(
        image, crop_size=crop, overlap=overlap, output_transform=fn, batch_size=2
    )
    for crop, source, destination in crop_iterator:
        for _crop, _source, _destination in zip(crop, source, destination):
            output[_destination] = _crop[_source]
            assert np.allclose(_crop[_source], image[_destination] * 100 - 3)


def test_batched_source_destination_agreement_with_pad():
    image = np.random.randn(20, 10, 10)
    output = np.empty_like(image)
    crop = (10, 10, 10)
    overlap = [2, 2, 2]

    def fn(x):
        return x * 100 - 3

    crop_iterator = Slicer(
        image,
        crop_size=crop,
        overlap=overlap,
        output_transform=fn,
        batch_size=2,
        pad=True,
    )
    for crop, source, destination in crop_iterator:
        for _crop, _source, _destination in zip(crop, source, destination):
            output[_destination] = _crop[_source]

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
