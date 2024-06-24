import numpy as np
from PIL import Image
from numpy import ndarray
import cv2


def getImageFileSize(file: str):
    img = Image.open(file)
    return img.size


def imRead(file: str, targetHeight: int = 0) -> ndarray:
    """
    Read and scale images based on targetHeight.
    Return: 3-channel image
    """
    img = Image.open(file)
    if targetHeight:
        ratio = (targetHeight / float(img.size[1]))
        wSize = int((float(img.size[0]) * float(ratio)))
        img = img.resize((wSize, targetHeight), Image.ANTIALIAS)
    npImg = np.array(img)
    if len(npImg.shape) == 2:
        npImg = np.stack((npImg,) * 3, axis=-1)
    return npImg


def drawBBoxesOnImage(image: np.ndarray, bboxes, scaleFactor: float = 1.) -> np.ndarray:
    r"""scaleFactor stands for (display size/image size)"""
    for label in bboxes:
        x1, y1, x2, y2 = [int(i * scaleFactor + 0.5) for i in label["bbox"]]
        cv2.rectangle(img=image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255))
    return image


def heatmapColormap(image: np.ndarray, code) -> np.ndarray:
    r"""The Input Image should be single channel 0-1 float.
    The output is a 3-channel BGR(Not RGB) 0-255 uint8 image.
    """
    return cv2.applyColorMap((image * 255).astype(np.uint8), code)


def superimposeHeatmapToImage(heatmap: np.ndarray, image: np.ndarray) -> np.ndarray:
    assert heatmap.shape[:2] == image.shape[:2]
    # if len(image.sha)
    return cv2.addWeighted(image, 0.4, heatmap, 0.6, 0)


def pointToHeatmap(pointList, gaussianSize=99, normalize=True, heatmapShape=(800, 800)):
    canvas = np.zeros(heatmapShape)
    for p in pointList:
        if p[1] <= heatmapShape[0] and p[0] <= heatmapShape[1]:
            canvas[p[1]][p[0]] = 1
    g = cv2.GaussianBlur(canvas, (gaussianSize, gaussianSize), 0, 0)
    if normalize:
        g = cv2.normalize(g, None, alpha=0, beta=1,
                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return heatmapColormap(g, cv2.COLORMAP_JET)
