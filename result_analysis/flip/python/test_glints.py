import os
import numpy as np
import cv2
from PIL import Image
import flip

TEST_RGB_DOMAIN = True
TEST_FOURIER_DOMAIN = True
NUM_TEST_IMAGES = 200

def normalise(image):
    # Normalise the image to [0, 1]
    return image / 255.0

def compute_fft(image):
    # Apply FFT
    f = np.fft.fft2(image)
    # Shift the zero frequency component to the center
    fshift = np.fft.fftshift(f)
    # Compute the magnitude spectrum
    magnitude_spectrum = np.abs(fshift)
    return magnitude_spectrum.astype(np.float32)

def make_rgb_from_gray(image):
    return np.stack((image,) * 3, axis=-1)

if __name__ == '__main__':
    imgs_root = os.path.join('..', '..', 'data', 'bob_clarens')

    ref_rgb_errs = []
    gsir_rgb_errs = []
    tensoir_rgb_errs = []
    gtv_rgb_errs = []
    mine_rgb_errs = []

    ref_fft_errs = []
    gsir_fft_errs = []
    tensoir_fft_errs = []
    gtv_fft_errs = []
    mine_fft_errs = []

    for i in range(NUM_TEST_IMAGES):
        ref_path = os.path.join(imgs_root, 'ref', f'r_{i}.png')
        gsir_path = os.path.join(imgs_root, 'gsir', f'r_{i}.png')
        tensoir_path = os.path.join(imgs_root, 'tensoir', f'r_{i}.png')
        gtv_path = os.path.join(imgs_root, 'mine_gt', f'r_{i}.png')
        mine_path = os.path.join(imgs_root, 'mine', f'r_{i}.png')

        if TEST_RGB_DOMAIN:
            print(f'Processing images with index {i} in the RGB domain...')
            # Need to normalise the RGB to [0, 1] and use LDR mode, otherwise the errs will all be 0.
            ref_rgb = normalise(np.array(Image.open(ref_path).convert('RGB')))
            gsir_rgb = normalise(np.array(Image.open(gsir_path).convert('RGB')))
            tensoir_rgb = normalise(np.array(Image.open(tensoir_path).convert('RGB')))
            gtv_rgb = normalise(np.array(Image.open(gtv_path).convert('RGB')))
            mine_rgb = normalise(np.array(Image.open(mine_path).convert('RGB')))

            _, mean_rgb_err_0, _ = flip.evaluate(ref_rgb, ref_rgb, "LDR")
            _, mean_rgb_err_1, _ = flip.evaluate(ref_rgb, gsir_rgb, "LDR")
            _, mean_rgb_err_2, _ = flip.evaluate(ref_rgb, tensoir_rgb, "LDR")
            _, mean_rgb_err_3, _ = flip.evaluate(ref_rgb, gtv_rgb, "LDR")
            _, mean_rgb_err_4, _ = flip.evaluate(ref_rgb, mine_rgb, "LDR")

            ref_rgb_errs.append(mean_rgb_err_0)
            gsir_rgb_errs.append(mean_rgb_err_1)
            tensoir_rgb_errs.append(mean_rgb_err_2)
            gtv_rgb_errs.append(mean_rgb_err_3)
            mine_rgb_errs.append(mean_rgb_err_4)
        
        if TEST_FOURIER_DOMAIN:
            print(f'Processing images with index {i} in the Fourier domain...')
            ref_gray = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
            gsir_gray = cv2.imread(gsir_path, cv2.IMREAD_GRAYSCALE)
            tensoir_gray = cv2.imread(tensoir_path, cv2.IMREAD_GRAYSCALE)
            gtv_gray = cv2.imread(gtv_path, cv2.IMREAD_GRAYSCALE)
            mine_gray = cv2.imread(mine_path, cv2.IMREAD_GRAYSCALE)

            ref_spectrum = make_rgb_from_gray(compute_fft(ref_gray))
            gsir_spectrum = make_rgb_from_gray(compute_fft(gsir_gray))
            tensoir_spectrum = make_rgb_from_gray(compute_fft(tensoir_gray))
            gtv_spectrum = make_rgb_from_gray(compute_fft(gtv_gray))
            mine_spectrum = make_rgb_from_gray(compute_fft(mine_gray))

            _, mean_fft_err_0, _ = flip.evaluate(ref_spectrum, ref_spectrum, "HDR")
            _, mean_fft_err_1, _ = flip.evaluate(ref_spectrum, gsir_spectrum, "HDR")
            _, mean_fft_err_2, _ = flip.evaluate(ref_spectrum, tensoir_spectrum, "HDR")
            _, mean_fft_err_3, _ = flip.evaluate(ref_spectrum, gtv_spectrum, "HDR")
            _, mean_fft_err_4, _ = flip.evaluate(ref_spectrum, mine_spectrum, "HDR")

            ref_fft_errs.append(mean_fft_err_0)
            gsir_fft_errs.append(mean_fft_err_1)
            tensoir_fft_errs.append(mean_fft_err_2)
            gtv_fft_errs.append(mean_fft_err_3)
            mine_fft_errs.append(mean_fft_err_4)
    
    print('FLIP results:')
    print('================= RGB =================')
    print(f'Sanity check for RGB: {np.mean(ref_rgb_errs)}')
    print(f'Mean FLIP error for GS-IR (RGB): {np.mean(gsir_rgb_errs)}')
    print(f'Mean FLIP error for TensorIR (RGB): {np.mean(tensoir_rgb_errs)}')
    print(f'Mean FLIP error for GtV (RGB): {np.mean(gtv_rgb_errs)}')
    print(f'Mean FLIP error for Mine (RGB): {np.mean(mine_rgb_errs)}')
    print('================= Fourier =================')
    print(f'Sanity check for Fourier: {np.mean(ref_fft_errs)}')
    print(f'Mean FLIP error for GS-IR (Fourier): {np.mean(gsir_fft_errs)}')
    print(f'Mean FLIP error for TensorIR (Fourier): {np.mean(tensoir_fft_errs)}')
    print(f'Mean FLIP error for GtV (Fourier): {np.mean(gtv_fft_errs)}')
    print(f'Mean FLIP error for Mine (Fourier): {np.mean(mine_fft_errs)}')


