# Equirectangle Images

### Panorama v.s. Equirectangular Image

**Equirectangular/Lat-Long Image** is a specific projection commonly used to store a full spherical 360°×180° panorama, where longitude maps linearly to the x-axis and latitude maps linearly to the y-axis (**typically yielding a 2:1 aspect ratio**). 

**Panorama** is a broad term for any image that captures a wider-than-normal field of view. Usually a 360° spherical panorama is stored as an equirectangular image, but not all panoramas are equirectangular! (e.g., your photo may be able to take partial cylindrical panorama, which is not an equirectangular image.)

**Environment maps** are also typically stored as equirectangular images for 360° lighting, but they can also be saved in other projections like a cubemap. They are usually HDR images, stored in `.hdr` or `.exr` format to preserve float32 precision.

### Equirect Projection

We can project equirect images into normal images with **arbitrary camera intrinsics**, which is useful for curating camera datasets.

### API
.. automodule:: kiui.equirect
   :members: