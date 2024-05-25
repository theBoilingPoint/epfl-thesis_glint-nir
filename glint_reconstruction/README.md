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

Note that to run the code, you need to have a GPU with CUDA support. The code is tested on a NVIDIA GeForce RTX 2080 Ti with __CUDA 11.7__. Please also make sure to install the corresponding Pytorch version.

This directory also contains a `requirements.txt` file as reference in case you are unsure which package version to use. You don't need to install all the packages in the file. 

## How to Train
### Prepare the Dataset
The data used in this project follows the [nerf](https://github.com/bmild/nerf) format.

1. Create a folder called `data` in the root folder.
2. Click this [link](https://drive.google.com/drive/folders/1IxgRXdkViqAqc3cbfPNlXiHc7x9WIfUQ?usp=sharing) to download the pregenerated data. Inside each innermost folder, you only need the keep the `test` and `train` folder and `transforms_train.json` and `transforms_test.json` file to run the code. The rest of the files are used for reference. 
3. Put a desired dataset, for example `bob_clarens` into the `data` folder. The structure of the `data` folder should look like this:
    ```
    .
    └── data
        └── bob_clarens
            ├── test
            ├── train
            ├── transforms_test.json
            └── transforms_train.json
    ```
4. In the `configs` folder, edit `nerf_glints.json` to set the training parameters. Make sure that `ref_mesh` matches the path of your desired dataset. For example, for the dataset mentioned above, the path should be `data/bob_clarens`.

### Prepare the Code
1. To edit the occlusion effect, change the `alpha`, `beta`, and `gamma` parameters in the `shade()` function of `light.py` under the `render` folder.
2. To edit the glint effect, adjust `_ScreenSpaceScale`, `targetNDF`, and `maxNDF` in the constructor of the `GlintBRDF` class in `glint_brdf.py` under the `glint_brdf` folder.

### Run the Code
Run the following command to train the model:
```
python train.py --config configs/nerf_glints.json
```

## How to Test
If you test a saved model, you can run the `test_glint.py` script in the root folder. This step assumes that you have already trained and obtained saved models of `geometry`, `material`, and `light`. These models are saved in the output folder of the training step.

### Edit the Code
1. Create a folder to put the saved model. For example, create a folder called `out/final/bob` in the root folder.
2. Move the saved models to the folder you just created.  
3. Edit the script following the comments. You only need to edit the main function.

### Run the Code
Simply run
```
python test_glint.py
```


## Acknowledgement

This project makes use of the codes from [nvdiffrec](https://github.com/NVlabs/nvdiffrec) and [Deliot et al.](https://thomasdeliot.wixsite.com/blog/single-post/hpg23-real-time-rendering-of-glinty-appearance-using-distributed-binomial-laws-on-anisotropic-grids). Great thanks to their amazing work!