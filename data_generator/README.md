# Data Generation

This folder contains the code to generate the data used in the paper. The data is generated using the [Nori](https://wjakob.github.io/nori/) framework. Inside Nori, the glint BRDF uses the method proposed by [Atanasov et al.](https://doi.org/10.1145/2897839.2927391). 

The folder structure is shown below:
```
.
├── env_maps
├── meshes
│   ├── nerf
│   └── test_glint_num
└── scripts
    └── camera_poses
```

## How to Generate Data
### Generate Raw Data
If you are starting from scratch, set `POST_PROCESSING_TRAIN` and `POST_PROCESSING_TEST` to `False`.

Then, if you want to generate training data, set 
```
GENERATE_TRAINING_DATA = True
```
If you want to generate testing data, set 
```
GENERATE_TESTING_DATA = True
```

Click the button to run all cells in the notebook. It might take a few hours to generate all data.

`integrator_type` defines the type of data. Both `mask` and `image` must be generated before they can be processed. 

### Process Raw Data
After all masks and images are generated, if you want to process the training data, set 
```
POST_PROCESSING_TRAIN = True

```
If you want to process the testing data, set 
```
POST_PROCESSING_TEST = True
```
Since the masks generated in the previous step are not binary, `black_pixel_threshold` defines the threshold for a pixel to be considered black. `black_pixel_threshold` should be in range [0, 1]. The higher the threshold, the brighter a black pixel is.

Then again, click the button to run all cells in the notebook.

## Download Data
Click this [link](https://drive.google.com/drive/folders/1IxgRXdkViqAqc3cbfPNlXiHc7x9WIfUQ?usp=sharing) to download the pregenerated data. 