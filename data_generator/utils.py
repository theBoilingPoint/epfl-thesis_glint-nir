import numpy as np
import json
from PIL import Image
import os
import shutil
import matplotlib.pyplot as plt
import imageio


# Getting Samples ####################
######################################
def get_points_from_nerf(json_file):
    mat = []
    with open(json_file) as fp:
        cam_dict = json.load(fp)
        for frame in cam_dict["frames"]:
            mat.append((np.array(frame["transform_matrix"])[:, 3][:-1]))

    return mat


def visualise_points(samples):
    normalised_samples = samples / np.linalg.norm(samples, axis=1, keepdims=True)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        normalised_samples[:, 0], normalised_samples[:, 1], normalised_samples[:, 2]
    )
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    plt.xlabel("X-Axis", fontsize=14, fontweight="bold", color="red")
    plt.ylabel("Y-Axis", fontsize=14, fontweight="bold", color="green")
    plt.show()


# Camera Poses #######################
######################################
def rotate_around_x(vector, angle):
    # Define the rotation matrix
    theta = np.deg2rad(angle)
    # rotation around x axis for 90 degrees
    rot_matrix = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )

    return np.dot(rot_matrix, vector)


def rotate_around_y(vector, angle):
    # Define the rotation matrix
    theta = np.deg2rad(angle)
    # rotation around x axis for 90 degrees
    rot_matrix = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )

    return np.dot(rot_matrix, vector)


def rotate_around_z(vector, angle):
    # Define the rotation matrix
    theta = np.deg2rad(angle)
    # rotation around x axis for 90 degrees
    rot_matrix = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )

    return np.dot(rot_matrix, vector)


# Post Processing ####################
######################################
def correct_pixel_colour(path):
    mask = []
    img = Image.open(path)

    for item in list(img.getdata()):
        r, g, b = item

        if (
            r == 254 and g == 254 and b == 254
        ):  # apparently the pixels are not completely white
            r = g = b = 255

        mask.append((r, g, b))

    img.putdata(mask)
    img.save(path, "PNG")


def replace_masks(mask_dir):
    imgs_mask_paths = [img for img in os.listdir(mask_dir) if img.endswith("png")]
    for each_img in imgs_mask_paths:
        print(f"Correcting {each_img}...")
        correct_pixel_colour(f"{mask_dir}/{each_img}")

def process_rendered_results_with_mask(input_dir, out_dir, mode, integrator_type, img_format, threshold):
    # check threshold
    assert threshold >= 0 and threshold <= 1

    img_folder = os.path.join(input_dir, mode, integrator_type, img_format)
    mask_folder = os.path.join(input_dir, mode, "mask", "png")

    cur_working_dir = os.getcwd()
    for img_name in os.listdir(img_folder):
        print(f"Processing {mode} image {img_name}...")
        if not (img_name.endswith(".png") or img_name.endswith(".exr")):
            continue

        # load image and mask
        img_path = os.path.join(img_folder, img_name)
        mask_path = os.path.join(mask_folder, f"{img_name[:-4]}.png")

        img = np.array(imageio.v3.imread(img_path))
        mask = np.array(imageio.v3.imread(mask_path))

        # "black pixel" means anything that is very close to black (not necessarily [0,0,0])
        # this is a simple way to get rid of the background
        updated_threshold = threshold * 255
        black_pixel_mask = np.all(
            mask[..., :3] <= [updated_threshold, updated_threshold, updated_threshold],
            axis=-1,
        )
        mask[black_pixel_mask] = [0, 0, 0]
        mask[~black_pixel_mask] = [1, 1, 1]
        assert not np.all(mask == 0)

        # This will produces a result where the background is black [0,0,0] and the image is the original colour
        masked_img = img * mask

        # append an alpha channel
        alpha_value = 1
        if img_format == "png":
            alpha_value = 255
        alpha_channel = np.full(
            (img.shape[0], img.shape[1], 1), alpha_value, dtype=np.uint8
        )
        masked_img = np.concatenate((masked_img, alpha_channel), axis=-1)

        # set the alpha channel to 0 for all black pixels
        masked_img[(masked_img[..., :3] == [0, 0, 0]).all(axis=-1), 3] = 0

        # save the result to the root directory
        imageio.v3.imwrite(img_name, masked_img)

        tmp_dir = os.path.join(out_dir, mode, integrator_type)
        os.makedirs(tmp_dir, exist_ok=True)

        shutil.move(os.path.join(cur_working_dir, img_name), tmp_dir)
