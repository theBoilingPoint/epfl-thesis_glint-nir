# Pytorch Implementation of Glint Reconstruction

The structure of the folder is shown below:
```
.
├── configs
├── dataset
├── docker
├── geometry
├── images
└── render
    └── renderutils
        ├── c_src
        ├── glint_brdf
        │   └── noise_maps
        └── tests
```
The glint sampling method is implemented in folder `glint_brdf`.

## Acknowledgement

This project makes use of the codes from [nvdiffrec](https://github.com/NVlabs/nvdiffrec) and [Deliot et al.](https://thomasdeliot.wixsite.com/blog/single-post/hpg23-real-time-rendering-of-glinty-appearance-using-distributed-binomial-laws-on-anisotropic-grids). Great thanks to their amazing work!