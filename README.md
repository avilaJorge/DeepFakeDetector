# Deep Fake Detector 

A deep fake detector based on this [paper](https://storage.googleapis.com/img-hosting-bucket/unmasking.pdf).  The code in this paper was rewritten in PyTorch and tested on StyleGAN and StyleGAN2 images.

StyleGAN and StyleGAN2 images were generated using the code from the following repos.

[StyleGAN](https://github.com/NVlabs/stylegan)
[StyleGAN2](https://github.com/NVlabs/stylegan2)

## Pre-Processed Datasets

***Faces_HQ.hdf5***
- Contains pre-processed images from [paper dataset](https://storage.googleapis.com/img-hosting-bucket/unmasking.pdf)
- https://storage.googleapis.com/img-hosting-bucket/Faces_HQ.hdf5

*The following datasets contain images generated by StyleGAN2 for cars, cats and churches and StyleGAN1 for bedrooms.  All real images are from LSUN.*

*Pre-processing on cars results in a 1x363 vector while all other datasets used result in a 1x182 vector.*

***LSUN.hdf5*** 
- Includes cats, cars, and churches with padding on the pre-processed vectors of cats and churches.  
- This dataset was created to test the model on images of varying sizes.
- https://storage.googleapis.com/img-hosting-bucket/LSUN.hdf5

***LSUN_BCC_256.hdf5***
- Dataset containing only images of size ![formula](https://render.githubusercontent.com/render/math?math=256^2)
- Includes cats, churches, and bedrooms.
- https://storage.googleapis.com/img-hosting-bucket/LSUN_BCC_256.hdf5

***LSUN_256.hdf5***
- Dataset containing only images of size ![formula](https://render.githubusercontent.com/render/math?math=256^2)
- Includes cats and churches.
- https://storage.googleapis.com/img-hosting-bucket/LSUN_256.hdf5

***LSUN_Cars.hdf5***
- Dataset containing only cars of size ![formula](https://render.githubusercontent.com/render/math?math=512^2)
- https://storage.googleapis.com/img-hosting-bucket/LSUN_Cars.hdf5

***LSUN_BCCC_256.hdf5***
- Dataset containing all four datasets of size 256x256.  
- StyleGAN2 Cars is cropped and LSUN cars is resized.
- https://storage.googleapis.com/img-hosting-bucket/LSUN_BCCC_256.hdf5

![Collage](https://storage.googleapis.com/img-hosting-bucket/collage.jpg)

![Mean & Std.](https://storage.googleapis.com/img-hosting-bucket/mean_plt_bccc.jpg)