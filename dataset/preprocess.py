import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import measure
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def preprocess(img_path, target_size=(224, 224), padding=5):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"No image in path: {img_path}")

    _, thresh = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)

    # generate a map where each non-zero pixel is replaced with distance to the nearest 0 pixel
    distance = ndi.distance_transform_edt(thresh)

    #based on the distance map find the peaks in 50 by 50 rectangle
    coords = peak_local_max(distance, footprint=np.ones((50, 50)), labels=thresh)
    mask_peaks = np.zeros(distance.shape, dtype=bool)
    mask_peaks[tuple(coords.T)] = True
    markers = ndi.label(mask_peaks)[0]

    # Watershed segmentation
    labels = watershed(-distance, markers, mask=thresh)

    # Find the largest region
    regions = measure.regionprops(labels)
    if regions:
        largest_region = max(regions, key=lambda r: r.area)
        mask = np.zeros_like(img)
        mask[labels == largest_region.label] = 255

        img = cv2.bitwise_and(img, img, mask=mask)

        # Crop with padding
        minr, minc, maxr, maxc = largest_region.bbox
        y_new = max(0, minr - padding)
        x_new = max(0, minc - padding)
        y_end = min(img.shape[0], maxr + padding)
        x_end = min(img.shape[1], maxc + padding)

        img = img[y_new:y_end, x_new:x_end]

    if np.sum(img[:, img.shape[1] // 2:]) > np.sum(img[:, :img.shape[1] // 2]):
        img = cv2.flip(img, 1)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    img = cv2.resize(img, target_size)

    return img

