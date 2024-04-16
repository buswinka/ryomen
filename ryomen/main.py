from __future__ import annotations
from typing import (
    Sequence,
    Tuple,
    Callable,
    List,
    Any,
    Iterator,
    Generator,
    Union,
)
from copy import copy

Shape = Sequence[int]
Index = Tuple[Union[type(Ellipsis), slice]]

class TensorLike:
    """ generic class for any tensor like object: zarr, numpy, pytorch, etc... """

    shape: Shape

    def __init__(self):
        ...

    def __getitem__(self, item: Any) -> TensorLike:
        ...

    def size(self) -> Shape:
        ...


def _get_next_slice(
    ind: List[int], c: List[int], o: List[int], shape: Shape
) -> Tuple[Index, List[int]]:
    """
    given a current index ind, a crop c, overlap o, and image shape
    calculate the next index of a crop such that all tiles of the image are eventually
    covered

    :param ind: list of the current coordinate pos
    :param c: crop size
    :param o: overlap size
    :param shape: shape of ultimate image
    :return: Slice of
    """

    indices = [Ellipsis]
    for i in range(len(shape)):

        _x = ind[i] if ind[i] + c[i] <= shape[i] else shape[i] - c[i]
        indices.append(slice(_x, _x + c[i], 1))

    i = 0
    while i < len(shape):
        if ind[i] + c[i] - o[i] * 2 <= shape[i]:
            ind[i] += +c[i] - o[i] * 2
            i = float("inf")
        else:
            ind[i] = 0
            i += 1
    return tuple(indices), ind


def _nd_generator(crop, overlap, shape) -> Generator[Tuple[Index, Sequence[int]]]:
    """ so cursed """

    assert len(crop) == len(overlap)

    for c, o in zip(crop, overlap):
        if not c > o * 2:
            raise ValueError("Crop Size must be larger than overlap * 2")

    x = [0 for _ in crop]
    assert len(shape) >= len(crop)
    shape = copy(list(shape))

    # Get to identical shape
    while len(shape) > len(crop):
        shape.pop(0)

    ind, x = _get_next_slice(x, crop, overlap, shape)
    while not all((a + c) > b for a, b, c, o in zip(x, shape, crop, overlap)):
        yield ind, x
        ind, x = _get_next_slice(x, crop, overlap, shape)

    yield ind, x

    ind, x = _get_next_slice(x, crop, overlap, shape)
    yield ind, x


class Slicer:
    def __init__(
        self,
        image: TensorLike,
        crop_size: Sequence[int],
        overlap: Sequence[int],
        batch_size: int = 1,
        output_transform: Callable[[TensorLike], TensorLike] = lambda x: x,
        collate: Callable[
            [Sequence[TensorLike]], TensorLike | List[TensorLike]
        ] = lambda x: x,
        progress_bar: Callable[[Iterator | Generator], Any] = lambda x: x,
    ):
        """
        BioCropper is a generic cropping utility for separating up large microscopy images into smaller,
        equal sized, sub crops. It works on any array type as long as that array has the methods __getitem__(), size(),
        and a class variable: size. Tested Array types include: numpy.ndarray, torch.Tensor, zarr.Array


        Basic Usage:

        >>> import numpy as np
        >>> from ryomen import Slicer
        >>> image = np.random.randn((3, 100, 100, 100))  # A large, 3D array with a color channel
        >>> # image.shape is [B, C, X, Y, Z]
        >>> crop_size, overlap = (10, 10, 10), (2, 2, 2)  # we want to crop on the spatial dimensions!
        >>>
        >>> for crop, index = Slicer(image, crop_size, overlap)
        >>>     print(crop.shape)  # [3, 10, 10, 10]

        The function also allows for batching and a custom colate function, usefull if you want to use this
        for machine learning evaluation. The default behavior is to agregate all batched images into a list. A
        custom collate function should accept a list of arrays and may return anything.

        The default behavior looks like this...

        >>> import numpy as np
        >>> from ryomen import Slicer
        >>> image = np.random.randn((3, 100, 100, 100))  # A large, 3D array with a color channel
        >>> crop_size, overlap = (10, 10, 10), (2, 2, 2)  # we want to crop on the spatial dimensions!
        >>> for crop, index = Slicer(image, crop_size, overlap, batch_size=8)
        >>>     print(len(crop))  # 8

        Now with a custom colate function...

        >>> import numpy as np
        >>> from ryomen import Slicer
        >>> image = np.random.randn((3, 100, 100, 100))  # A large, 3D array with a color channel
        >>> crop_size, overlap = (10, 10, 10), (2, 2, 2)  # we want to crop on the spatial dimensions!
        >>> collate = lambda x: np.stack(x, axis=0)
        >>> for crop, index = Slicer(image, crop_size, overlap, batch_size=8, collate=collate)
        >>>     print(crop.shape)  # [8, 3, 10, 10, 10]

        This utility also allows for the evaluation of arbitrary functions on each crop prior to collating. This
        is useful if you want to normalize each crop. This function should accept an ArrayLike and return an ArrayLike

        >>> import numpy as np
        >>> from ryomen import Slicer
        >>> image = np.random.randn((3, 100, 100, 100))  # A large, 3D array with a color channel
        >>> crop_size, overlap = (10, 10, 10), (2, 2, 2)  # we want to crop on the spatial dimensions!
        >>> collate = lambda x: np.stack(x, axis=0)
        >>> normalize = lambda x: x / 255 * 2 - 1
        >>> for crop, index = Slicer(image, crop_size, overlap, batch_size=8, collate=collate, output_transform=normalize)
        >>>     print(crop.shape)  # [8, 3, 10, 10, 10]

        :param image: N Dimensional Array of any type
        :param crop_size: Tuple ints for cropping dimensions
        :param overlap: Tuple of ints for the overlap between crops
        :param batch_size: batch size of crops. Allows for returning multiple crops in one iteration
        :param output_transform: function to apply to each crop before returning
        :param collate: function to collate images, the default behavior is to return a list of crops
        :param progress_bar: function which wraps the iteration and displays a progress bar. (e.g. tqdm)
        """

        self.__N = None
        self.__image: TensorLike = image
        self.__crop_size = list(crop_size)
        self.__overlap = list(overlap)
        self.__batch_size = batch_size
        self.__output_fn = output_transform
        self.__collate_fn = collate
        self.__progress_bar = progress_bar

        # Check that all of the nasty inputs are handled...
        self._check_validity()

        # Adjust crop size such that we can return images smaller than crop size.
        # This is silent and might be confusing. The user should never have to access the crop size after creation.
        image_shape = self.__image.shape
        for i, size in enumerate(self.__crop_size):
            self.__crop_size[i] = (
                self.__crop_size[i]
                if self.__crop_size[i]
                < image_shape[i + len(image_shape) - len(self.__crop_size)]
                else image_shape[i + len(image_shape) - len(self.__crop_size)]
            )


    def _check_validity(self):
        """ simply checks the validity of all inputs """

        if not hasattr(self.__image, 'size'):
            raise ValueError(f'Input array of type {type(self.__image)} does not have the method: size')
        if not hasattr(self.__image, '__getitem__'):
            raise ValueError(f'Input array of type {type(self.__image)} does not have the method: __getitem__')
        if not hasattr(self.__image, 'shape'):
            raise ValueError(f'Input array of type {type(self.__image)} does not have the method: shape')

        shape: Shape = self.__image.shape

        # Check if the crop size dimension is reasonable
        if len(shape) < len(self.__crop_size):
            raise ValueError(
                f"image of shape: {shape} cannot be subdivided into crops of shape: {self.__crop_size} with overlap: {self.__overlap}"
            )

        # Check if the overlap size dimension is reasonable
        if len(shape) < len(self.__overlap):
            raise ValueError(
                f"image of shape: {shape} cannot be subdivided into crops of shape: {self.__crop_size} with overlap: {self.__overlap}"
            )

        if len(self.__crop_size) != len(self.__overlap):
            raise ValueError(f"crop_size and overlap_size must have the same length")

        for d, (c, o) in enumerate(zip(self.__crop_size, self.__overlap)):
            if c - (o * 2) < 0:
                raise ValueError(
                    f"Overlap in {d} dimension cannot be equal to or larger than crop size... {c=} - {o*2=} = {c - (o * 2)} < {c}"
                )

            if c <= 0:
                raise ValueError(f"crop size must not be negative")
            if o < 0:
                raise ValueError(f"Overlap must not be negative")

            if not isinstance(c, int):
                raise TypeError(
                    f"Crop sizes must be an integer, not {c=} of type {type(c)}"
                )

            if not isinstance(o, int):
                raise TypeError(
                    f"Overlap sizes must be an integer, not {o=} of type {type(o)}"
                )

        if not isinstance(self.__batch_size, int):
            raise TypeError(f"Batch sizes must be a positive integer > 0")

        if self.__batch_size <= 0:
            raise ValueError(f"Batch sizes must be a positive integer > 0")

        n = len(
            list(_nd_generator(self.__crop_size, self.__overlap, self.__image.shape))
        )
        if self.__batch_size > n:
            raise RuntimeError(
                "requested batch size is greater than the entirety of the image cropped. "
                f"All crops collated would result in {n} batches. {n} < {self.__batch_size}"
            )

    def __iter__(self) -> Tuple[TensorLike | List[TensorLike], Index, Index]:
        image_shape: Shape = self.__image.shape  # C, X, Y, Z

        output_cache = []
        for ind, _ in self.__progress_bar(
            _nd_generator(self.__crop_size, self.__overlap, image_shape)
        ):

            # _nd_generator returns a list of slices
            source = self._source()
            destination = self._destination(tuple(a.start for a in ind if isinstance(a, slice)), image_shape)
            crop = self.__image[tuple(ind)]
            output_cache.append(
                (self.__output_fn(crop), source, destination)
            )

            # If we've added more to the output cache than the batch size, we flush the batch away
            if len(output_cache) == self.__batch_size:
                yield self._flush_output(output_cache)

        # Final Yield - we may have a smaller batch size than expected...
        if len(output_cache) > 0:
            yield self._flush_output(output_cache)

    def _flush_output(self, output_cache):
        """ pop off the cache and convert to images, sources, and destinations """
        output = []
        for i in range(len(output_cache)):
            output.append(output_cache.pop(0))

        images: List[TensorLike] | TensorLike = self.__collate_fn(
            [t for t, s, d in output]
        )
        sources: List[Index] = [tuple(s) for t, s, d in output]
        destinations: List[Index] = [tuple(d) for t, s, d in output]

        if len(images) == len(sources) == len(destinations) == 1:
            return images[0], sources[0], destinations[0]

        else:
            return images, sources, destinations

    def __len__(self):
        """ Returns the number of slices this iterator will iterate over """
        if self.__N is None:
            self.__N = len(
                list(
                    _nd_generator(self.__crop_size, self.__overlap, self.__image.shape)
                )
            )
        return self.__N

    def _source(self) -> Index:
        """ returns indices of the crop excluding overlap """
        source = [Ellipsis]
        for o, s in zip(self.__overlap, self.__crop_size):
            source += [slice(o, -o if o != 0 else s)]
        return tuple(source)

    def _destination(self, xyz: Sequence[int], shape: Shape) -> Index:
        """
        returns the indicies to place the non-overlaping region of the image into a new tensor
        """
        shape = list(shape)
        while len(shape) > len(self.__crop_size):
            shape.pop(0)

        destination: Sequence[Ellipsis | slice] = [Ellipsis]
        for a, c, o, s in zip(xyz, self.__crop_size, self.__overlap, shape):
            a = a if a + c <= s else s - c
            destination += [slice(a + o, a + c - o, 1)]

        return tuple(destination)


if __name__ == "__main__":
    import numpy as np
    image = np.random.randn(1, 10, 10, 10)
    output = np.empty_like(image)
    crop = (5, 5, 5)
    overlap = [0, 0, 0]
    
    crop_iterator = Slicer(image, crop_size=crop, overlap=overlap)
    for crop, source, destination in crop_iterator:
        output[destination] = crop[source]
    
    assert np.allclose(image, output)