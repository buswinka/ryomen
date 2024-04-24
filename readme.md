# Ryomen
## For slicing up large microscopy images

Ryomen is a lightweight, no dependency utility for working with large biological microscopy images. 


### Why do we need this?

Modern microscopes can image cells, tissue, or whatever else, over huge spatial areas, resulting in images that can be
quite large (hundreds of gigabytes). This can make applying image processing or deep learning algorithms difficult. 
Ryomen makes this easy by automatically slicing up large images and providing an easy-to-use interface for working with the
crops. 

### What can it do? 

Ryomen works with N dimensional, arbitrary array types, including numpy, zarr, pytorch, or anything else. It simplifies
the process of slicing large images, with slices that may overlap, applying a function to the image, and extracting the
non-overlaping subregions. First, install with pip: ```pip install ryomen```. The following example crops a large ```image``` into 512x512 crops overlapping by 64px, aplies a hard to run
function ```expensive_fn```, and takes the result and puts it into a pre-allocated ```output``` array.

```python
crop_size = (512, 512)
overlap = (64, 64)
slices = ryomen.Slicer(image, crop_size=crop_size, overlap=overlap)

for crop, source, destination in slices:
    # Run the expensive fn on a crop of the whole image
    crop = expensive_fn(crop)
    
    # Slot the crop into an output array, excluding the overlap
    output[destination] = crop[source]  
```

A full code example might look like this:

```python
import skimage.io as io
import numpy as np
import zarr
from ryomen import Slicer

image = io.imread("really_huge_8bit_2d_image.tif")  # Shape of [3, 10000, 10000] 
c, x, y = image.shape
output: zarr.Array = zarr.open(
    'really_huge_output_on_disk.zarr',
    mode="w",
    shape=(c, x, y),
    dtype="|f2",
)

expensive_fn = lambda x: x + 1  # might be a ML model, or anything else

crop_size = (512, 512)
overlap = (64, 64)
slices = Slicer(image, crop_size=crop_size, overlap=overlap)
for crop, source, destination in slices:
    # Run the expensive fn on a crop of the whole image
    crop = expensive_fn(crop)
    
    # Slot the crop into an output array, excluding the overlap
    output[destination] = crop[source]  

```

You can apply arbitrary functions to a crop after it has been extracted from the large image. This is useful for working 
with a large uint8 image that might need to be cast to float for a ML model. 

```python
import skimage.io as io
import numpy as np
import zarr
from ryomen import Slicer

image = io.imread("really_huge_8bit_3d_image.tif")  # Shape of [3, 10000, 10000, 500] 
c, x, y, z = image.shape
output: zarr.Array = zarr.open(
    'really_huge_output_on_disk.zarr',
    mode="w",
    shape=(c, x, y, z),
    dtype="|f2",
)
crop_size = (512, 512, 64)
overlap = (64, 64, 8)
# Output transform will be applied to each crop.
output_transform = lambda x: x.astype(float).div(255).sub(0.5).mul(2)  # cast to float an normalize 

slices = Slicer(image, crop_size=crop_size, overlap=overlap, output_transform=output_transform)

expensive_fn = lambda x: x + 1  # might be a ML model, or anything else

for crop, source, destination in slices:
    assert crop.max() <= 1  # True
    assert crop.dtype() == np.float32  # True
    
    # Run the expensive fn on a crop of the whole image
    crop = expensive_fn(crop)
    
    # Slot the crop into an output array, excluding the overlap
    output[destination] = crop[source]  
```

Finally, ryomen can automatically batch tiles if you wanted to run each through a ML model. By default, a list of
tensors will be returned, however, you may suply a custom collate fn to handle batched inputs. 

```python
import skimage.io as io
import torch
from torch import nn as nn
import zarr
from ryomen import Slicer

image = io.imread("really_huge_8bit_2d_image.tif")  # Shape of [3, 10000, 10000] 
c, x, y = image.shape
output: zarr.Array = zarr.open(
    'really_huge_output_on_disk.zarr',
    mode="w",
    shape=(c, x, y),
    dtype="|f2",
)
crop_size = (512, 512, 64)
overlap = (64, 64, 8)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64

# Output transform will be applied to each crop.
output_transform = lambda x: torch.from_numpy(x).to(device).div(255).sub(0.5).div(0.5)  # cast to cuda tensor an normalize 
collate = lambda x: torch.stack(x, dim=0)  # turns a list of tensors to one tensor

slices = Slicer(image, 
                crop_size=crop_size, 
                overlap=overlap, 
                output_transform=output_transform, 
                batch_size=64,
                collate=collate)

pytorch_model = lambda x: nn.Conv3d(3, 1, kernel_size=3)

for crop, source, destination in slices:
    assert crop.ndim == 4  # crop shape is -> [B=64, 3, 512, 512]
    
    # Run the expensive fn on a crop of the whole image
    crop = pytorch_model(crop)
    
    # Slot the batched crop into an output array, excluding the overlap
    for b, (_source, _destination) in enumerate(zip(source, destination)):
        output[_destination] = crop[b, ...][_source]  
```

### Change Log


#### 0.0.5
    - fixed bug with duplicate indices being shown, leading to redundant behaviro
    - added support for zarr arrays, as they do not implement a flip method or support negative indexing
    - refactored for readability

#### 0.0.4
    - padding now works with pure indexing, for reflections. Tested up to 3D with numpy. 


#### 0.0.3
    - First working version

Ryomen is titled after the first name of the main antagonist of JJK who slices up everything.