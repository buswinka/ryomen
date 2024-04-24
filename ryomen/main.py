from __future__ import annotations
from types import EllipsisType
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
from copy import copy, deepcopy

Shape = Sequence[int]
Index = Tuple[Union[type(Ellipsis), slice], ...]
SliceList = List[Union[type(Ellipsis), slice]]


class TensorLike:
    """ generic class for any tensor like object: zarr, numpy, pytorch, etc... """

    shape: Shape

    def __init__(self):
        ...

    def __getitem__(self, item: Any) -> TensorLike:
        ...

    def __setitem__(self, index: Any, item: Any) -> TensorLike:
        ...

    def __iter__(self) -> Iterator:
        ...

    def __next__(self) -> TensorLike:
        ...

    def size(self) -> Shape:
        ...

    def flip(self, dim: int) -> TensorLike:
        ...


def _get_next_slice(
        ind: List[int], c: List[int], o: List[int], shape: Shape, pad: bool
) -> Tuple[Index, List[int]]:
    """
    given a current index ind, a crop c, overlap o, and image shape
    calculate the next index of a crop such that all tiles of the image are eventually
    covered

    :param ind: list of the current coordinate pos
    :param c: crop size
    :param o: overlap size
    :param shape: shape of ultimate e
    :return: Slice of
    """

    indices: list[EllipsisType | slice] = [Ellipsis]
    for i in range(len(shape)):
        _x = ind[i] if ind[i] + c[i] <= shape[i] + (o[i] * pad) else shape[i] - c[i] + (o[i] * pad)
        indices.append(slice(_x, _x + c[i], 1))

    i = 0
    while i < len(shape):
        if ind[i] + c[i] - o[i] * 2 <= shape[i] + (o[i] * pad):
            ind[i] += c[i] - o[i] * 2
            break
        else:
            ind[i] = -o[i] if pad else 0
            i += 1
    return tuple(indices), ind


def _nd_generator(
        crop: List[int], overlap: List[int], shape: Shape, pad: bool
) -> Generator[Tuple[Index, Sequence[int]], None, None]:
    """ so cursed """
    assert len(crop) == len(overlap)

    # In the weird case where an overlap is larger than a crop, we should fail here...
    for c, o in zip(crop, overlap):
        if not c > o * 2:
            raise ValueError(f"Crop Size must be larger than overlap * 2. {c=} !> {o=} * 2")

    # init a list of zeros as the "first" crop location
    x = [-o if pad else 0 for _, o in zip(crop, overlap)]
    assert len(shape) >= len(crop)

    # We cache all previous indices here, as the cost of looking this up
    # is presumably less than the cost of re-running the expensive fn
    previously_yeilded = []

    # copy because lists are mutable and we need to mutate it in this scope but not the higher one...
    shape = copy(list(shape))

    # Leading dimensions are allowed, therefore we pop off all leading dimensions until the shape
    # and crop shape are the same!
    while len(shape) > len(crop):
        shape.pop(0)

    ind, x = _get_next_slice(x, crop, overlap, shape, pad)
    while not all((a + c) > b + (o * pad) for a, b, c, o in zip(x, shape, crop, overlap)):
        if str(ind) not in previously_yeilded:
            previously_yeilded.append(str(ind))
            yield ind, x

        ind, x = _get_next_slice(x, crop, overlap, shape, pad)

    if str(ind) not in previously_yeilded:
        previously_yeilded.append(str(ind))
        yield ind, x

    ind, x = _get_next_slice(x, crop, overlap, shape, pad)
    if str(ind) not in previously_yeilded:
        previously_yeilded.append(str(ind))
        yield ind, x


def default_colate(x: Sequence[TensorLike]) -> TensorLike | Sequence[TensorLike]:
    return x


def default_progress_bar(x: Iterator | Generator) -> Any:
    return x


class Slicer:
    def __init__(
            self,
            image: TensorLike,
            crop_size: Sequence[int],
            overlap: Sequence[int],
            batch_size: int = 1,
            pad: bool = False,
            output_transform: Callable[[TensorLike], TensorLike] = lambda x: x,
            collate: Callable[
                [Sequence[TensorLike]], TensorLike | Sequence[TensorLike]
            ] = default_colate,
            progress_bar: Callable[[Iterator | Generator], Any] = default_progress_bar,
    ):
        """
        Ryomen is a generic cropping utility for separating up large microscopy images into smaller,
        equal sized, sub crops.

        Ryomen works with any generic array type, as long as it implements the following methods: __getitem__,
        __setitem__, __iter__, and __size__. Furthermore, the array type must support either a flip method, or
        support negative indexing, and must implement a parameter shape. These criteria basically fit all
        numpy-like arrays. Tested array libraries include zarr, numpy, and pytorch.

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
        >>> for crop, index = Slicer(maige, crop_size, overlap, batch_size=8)
        >>>     print(len(crop))  # 8

        Now with a custom collate function...

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
        :param pad: pads the image by overlap if true. Only supported if input tensor implements a flip method.
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

        self.__can_pad = (
            False  # Default, is changed to value of pad after suitable checks
        )
        self.__support_negative_strides = (
            False  # Default, is changed to true after suitable checks
        )

        # Zarr arrays dont have a flip method, or support negative indices. However,
        # after one indexing, they turn into numpy arrays which DO support negative
        # indices. To keep no dependencies, we do not type check, however
        # if it looks like a duck...
        self.__probably_a_zarr_array = False

        # Check that all of the nasty inputs are handled...
        self._check_validity()

        # Adjust crop size such that we can return images smaller than crop size.
        # This is silent and might be confusing. The user should never have to access the crop size after creation.

        self.__user_defined_crop_larger_than_image = False
        image_shape = self.__image.shape

        for i, size in enumerate(self.__crop_size):
            if (
                    not self.__crop_size[i]
                        < image_shape[i + len(image_shape) - len(self.__crop_size)]
            ):
                self.__user_defined_crop_larger_than_image = True

            self.__crop_size[i] = (
                self.__crop_size[i]
                if self.__crop_size[i]
                   < image_shape[i + len(image_shape) - len(self.__crop_size)]
                else image_shape[i + len(image_shape) - len(self.__crop_size)]
            )

        # Here we check if we can pad the image with indexing. We require negative indexing support, an implement flip
        # method, or it to be a zarr array

        try:  # test for negative stride
            self.__image[0:1:-1, ...]  # can flip with negative indices
            self.__support_negative_strides = True
            self.__can_pad = pad

        except:
            try:
                self.__image[0:1, ...][::-1, ...]
                self.__probably_a_zarr_array = True
                self.__can_pad = pad

            except:
                if not hasattr(self.__image, "flip"):

                    raise RuntimeError(
                        "Your image array does not implement a flip method or support negative strides. Therefore, padding is not supported."
                    )
                else:
                    self.__can_pad = pad

    def _check_validity(self):
        """ simply checks the validity of all inputs """

        if not hasattr(self.__image, "size"):
            raise ValueError(
                f"Input array of type {type(self.__image)} does not have the method: size"
            )
        if not hasattr(self.__image, "__getitem__"):
            raise ValueError(
                f"Input array of type {type(self.__image)} does not have the method: __getitem__"
            )
        if not hasattr(self.__image, "shape"):
            raise ValueError(
                f"Input array of type {type(self.__image)} does not have the method: shape"
            )

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
            list(
                _nd_generator(
                    self.__crop_size, self.__overlap, self.__image.shape, self.__can_pad
                )
            )
        )
        if self.__batch_size > n:
            raise RuntimeError(
                "requested batch size is greater than the entirety of the image cropped. "
                f"All crops collated would result in {n} batches. {n} < {self.__batch_size}"
            )

    def __iter__(
            self,
    ) -> Generator[
        Tuple[
            TensorLike | Sequence[TensorLike],
            Index | Sequence[Index],
            Index | Sequence[Index],
        ],
        None,
        None,
    ]:
        image_shape: Shape = self.__image.shape  # C, X, Y, Z

        output_cache = []
        for ind, xyz in self.__progress_bar(
                _nd_generator(self.__crop_size, self.__overlap, image_shape, self.__can_pad)
        ):

            # _nd_generator returns a list of slices
            source = self._get_source_index()
            destination = self._get_destination_index(
                tuple(a.start for a in ind if isinstance(a, slice)), image_shape
            )
            crop = self._index_image_with_pad(tuple(ind))
            output_cache.append((self.__output_fn(crop), source, destination))

            # If we've added more to the output cache than the batch size, we flush the batch away
            if len(output_cache) == self.__batch_size:
                yield self._flush_output(output_cache)

        if len(output_cache) > 0:
            yield self._flush_output(output_cache)

    def _flush_output(
            self, output_cache
    ) -> Tuple[
        TensorLike | Sequence[TensorLike],
        Index | Sequence[Index],
        Index | Sequence[Index],
    ]:
        """ pop off the cache and convert to images, sources, and destinations """
        output = []
        for i in range(len(output_cache)):
            output.append(output_cache.pop(0))

        images: Sequence[TensorLike] | TensorLike = self.__collate_fn(
            [t for t, s, d in output]
        )

        sources: List[Index] = [tuple(s) for t, s, d in output]
        destinations: List[Index] = [tuple(d) for t, s, d in output]

        if (
                isinstance(images, list)
                and len(images) == len(sources) == len(destinations) == 1
        ):
            return images[0], sources[0], destinations[0]

        else:
            return images, sources, destinations

    def __len__(self):

        """ Returns the number of slices this iterator will iterate over """
        if self.__N is None:
            self.__N = len(
                list(
                    _nd_generator(
                        self.__crop_size,
                        self.__overlap,
                        self.__image.shape,
                        pad=self.__can_pad,
                    )
                )
            )
        return self.__N

    @staticmethod
    def minmax(a: slice, m: int) -> slice:

        if a.start < 0:
            delta = abs(a.start)
            a = slice(a.start + delta, a.stop + delta, a.step)
        elif a.stop > m:
            delta = abs(a.stop - m)
            a = slice(a.start - delta, a.stop - delta, a.step)

        return slice(min(max(a.start, 0), m), min(max(a.stop, 0), m), a.step)

    def _flip_array(self, array: TensorLike, index: int):
        """ handes flipping a zarr array """
        if self.__probably_a_zarr_array and self.__can_pad:
            ind = []
            for i, s in enumerate(array.shape):
                if i == index:
                    ind.append(slice(s, None, -1))
                else:
                    ind.append(slice(0, None, 1))

            return array[tuple(ind)]
        elif not self.__support_negative_strides:  # can flip...
            return array.flip(index)
        else:
            RuntimeError('unknown error. cannot flip or not a zarr array')

    def _index_image_with_pad(self, index: Index) -> TensorLike:
        """
        Terrible function to index the image, and through indexing alone, pads the image with reflections. It
        should do this in N Dimensions, but was an absolute nightmare to code, and there are so many edge cases. Ive
        tried my best to handle as many as possible, but innevitably some may fall through thte cracks.
        :param index:
        :return:
        """

        if not self.__can_pad:
            return self.__image[index]

        # axis is smaller (-1) or larger than size
        shape = copy(list(self.__image.shape))
        while len(shape) < len(self.__crop_size):
            shape.pop(0)  # remove leading dim

        # Remove all Ellipsis. There should only be one at position 0, however this
        # implementation makes linters happy...
        modified_index: List[slice] = [
            s for s in list(copy(index)) if isinstance(s, slice)
        ]

        # First we check if any padding is even necessary...
        padding_necessary = False
        for i, (ind, s, o) in enumerate(zip(modified_index, shape, self.__overlap)):
            if ind.start < 0:
                padding_necessary = True
            elif ind.stop > s:
                padding_necessary = True

        if not padding_necessary:
            return self.__image[index]

        # We have a problem! We need a library agnostic way to create a tensor and pop it in...
        # We can do this by indexing another part of the tensor, and doing a deepcopy. I think
        # this should work for basically everything. Unsure about zarr...
        # we can then slot the info into this output array
        _output_index, _ = _get_next_slice(
            [0 for _ in self.__crop_size],
            c=self.__crop_size,
            o=self.__overlap,
            shape=shape,
            pad=False,
        )

        output_array = (self.__image[_output_index] / float("inf")) + 1
        first_access = False
        n_leading_dimensions = len(shape) - len(self.__crop_size)
        for i, (ind, s, o) in enumerate(zip(modified_index, shape, self.__overlap)):
            pad = None
            other = None

            padding_source: SliceList = [Ellipsis]
            other_source: SliceList = [Ellipsis]
            padding_destination: SliceList = [Ellipsis]
            other_destination: SliceList = [Ellipsis]

            has_padded = False

            # Padding on the left
            if ind.start < 0:
                has_padded = True
                for d, (current_slice, c) in enumerate(
                        zip(modified_index, self.__crop_size)
                ):
                    x = c if first_access else shape[d + n_leading_dimensions]
                    if i + n_leading_dimensions != d:
                        padding_source.append(self.minmax(current_slice, c if first_access else x))
                        other_source.append(self.minmax(current_slice, c if first_access else x))
                        padding_destination.append(slice(0, c, 1))
                        other_destination.append(slice(0, c, 1))
                    else:
                        padding_source.append(
                            slice(abs(ind.start) - 1, None, -1)
                            if self.__support_negative_strides
                            else slice(0, abs(ind.start), 1)
                        )
                        padding_destination.append(slice(0, abs(ind.start), 1))
                        other_source.append(slice(0, ind.stop, 1))
                        other_destination.append(slice(abs(ind.start), None, 1))

            # Padding on the right
            elif ind.stop > s:
                has_padded = True
                for d, (current_slice, c) in enumerate(
                        zip(modified_index, self.__crop_size)
                ):
                    x = c if first_access else shape[d + n_leading_dimensions]
                    if i + len(shape) - len(self.__crop_size) != d:
                        padding_source.append(self.minmax(current_slice, x))
                        other_source.append(self.minmax(current_slice, x))
                        padding_destination.append(slice(0, c, 1))
                        other_destination.append(slice(0, c, 1))
                    else:
                        padding_source.append(
                            slice(x, -(ind.stop - s) - 1, -1)
                            if self.__support_negative_strides
                            else slice(-(ind.stop - s), x, 1)
                        )
                        other_source.append(slice(ind.stop - s if first_access else current_slice.start, x))
                        padding_destination.append(slice(-(ind.stop - s), c, 1))
                        other_destination.append(slice(0, -(ind.stop - s), 1))

            if has_padded:
                to_access = output_array if first_access else self.__image
                pad = deepcopy(to_access[tuple(padding_source)])
                if not self.__support_negative_strides:
                    pad = self._flip_array(pad, i + len(shape) - len(self.__crop_size))
                other = deepcopy(to_access[tuple(other_source)])

                output_array[tuple(other_destination)] = other
                output_array[tuple(padding_destination)] = pad
                first_access = True

        return output_array

    def _get_source_index(self) -> Index:
        """ returns indices of the crop excluding overlap """
        source = [Ellipsis]
        for o, s in zip(self.__overlap, self.__crop_size):
            source += [slice(o, -o if o != 0 else s)]
        return tuple(source)

    def _get_destination_index(self, xyz: Sequence[int], shape: Shape) -> Index:
        """
        returns the indicies necessary to place the non-overlaping region of the image into a new tensor
        """
        shape = list(shape)
        while len(shape) > len(self.__crop_size):
            shape.pop(0)

        destination: Sequence[EllipsisType | slice] = [Ellipsis]
        for a, c, o, s in zip(xyz, self.__crop_size, self.__overlap, shape):
            a = a if a + c < s + (self.__can_pad * o) else s - c + (self.__can_pad * o)
            destination += [slice(a + o, (a + c) - o, 1)]

        return tuple(destination)
