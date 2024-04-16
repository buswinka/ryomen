# Ryomen
## For slicing up large microscopy images

Ryomen is a lightweight, no dependancy utility for working with large biological microscopy images. 

### Why do we need this?

Modern microscopes can image cells, tissue, or whatever else, over huge spatial areas, resulting in images that can be
quite large (hundreds of gigabytes). This can make applying image processing or deep learning algorithms difficult. 
Ryomen makes this easy by automatically slicing up large images and proving a easy to use interface for working with the
crops. 

### What can it do? 

Ryomen works with N dimensional, arbitrary array types, including numpy, zarr, pytorch, or anything else. It simplifies
the process of slicing large images, with slices that may overlap, applying a function to the image, and extracting the
non-overlaping subregions. A simple use case might look like this

```python
import skimage.io as io
import zarr
from ryomen import Slicer

image = io.imread("really_huge_3d_images.tif")  # Shape of [3, 10000, 10000, 500] 
c, x, y, z = image.shape
output: zarr.Array = zarr.open(
    'really_huge_output_on_disk.zarr',
    mode="w",
    shape=(c, x, y, z),
    dtype="|f2",
)
crop_size = (512, 512, 64)
overlap = (64, 64, 8)
slices = Slicer(image, crop_size=crop_size, overlap=overlap)

expensive_fn = lambda x: x + 1  # might be a ML model, or anything else

for crop, source, destination in slices:
    # Run the expensive fn on a crop of the whole image
    crop = expensive_fn(crop)
    
    # Slot the crop into an output array, excluding the overlap
    output[destination] = crop[source]  
    



```