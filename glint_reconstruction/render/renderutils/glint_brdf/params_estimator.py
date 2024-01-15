import os
import numpy as np
import torch
import imageio
import cv2


def normalize_array(arr, min, max):
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized_array = min + ((arr - min_val) * (max - min)) / (max_val - min_val)
    return normalized_array


def f(img, dim):
    return (
        torch.max_pool2d(img, dim).repeat_interleave(dim, -1).repeat_interleave(dim, -2)
    )


def get_counts_from_folder(img_folder):
    glint_counts = []

    print("Getting glint counts from data in folder: ", img_folder)

    for img_name in os.listdir(img_folder):
        if img_name == ".DS_Store":
            continue
        img_path = os.path.join(img_folder, img_name)

        img = imageio.v3.imread(img_path).copy()
        img[img[..., -1] == 0] = [0.0, 0.0, 0.0, 1.0]
        img = img[..., :3]

        # convert exr to png according to https://stackoverflow.com/questions/50748084/convert-exr-to-jpeg-using-imageio-and-python
        img *= 65535
        img[img > 65535] = 65535
        low_dynamic_rgb_img = img.astype(np.uint16)
        # convert img to grey scale
        grey_img = normalize_array(
            0.2989 * low_dynamic_rgb_img[..., 0]
            + 0.5870 * low_dynamic_rgb_img[..., 1]
            + 0.1140 * low_dynamic_rgb_img[..., 2],
            0,
            1,
        )

        timg = torch.from_numpy(grey_img)
        timg[timg == 0] = 2
        f1 = -f(-timg[None,], 4)
        timg[timg == 2] = 0
        m = f1[0] < timg - 0.3

        analysis = cv2.connectedComponentsWithStats(m.byte().numpy(), 4, cv2.CV_32S)
        (totalLabels, label_ids, values, centroid) = analysis

        glint_counts.append(totalLabels)

    return glint_counts


def get_stats_of_obj(path):
    counts = []
    counts += get_counts_from_folder(os.path.join(path, "train"))
    counts += get_counts_from_folder(os.path.join(path, "test"))

    res = torch.log(torch.tensor(counts, dtype=torch.float32))

    mean = res.mean()
    std = res.std()
    print(
        f"Predicted Parameters -> _LogMicrofacetDensity: {mean}, _DensityRandomization: {std}"
    )
    return (mean, std)
