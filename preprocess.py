import cv2
import numpy as np
from scipy.interpolate import splprep, splev


def preprocess_images_masking(img_path, img_name_, white_background=False):
    """First step preprocessing of the image for digit detection.
    This function will mask out enclosed shapes in an image.
    Thus, enclosed shapes will be detected and rest of image will have black background.

    Args:
        img_path (str): path of the image for preprocessing and masking

        img_name_ (str): name of the image for preprocessing and masking

        white_background (bool): if True, background whitening will be applied for image


    Returns:
        str: output path of the preprocessed image
    """

    # load image
    img = cv2.imread(img_path)

    ## remove background
    # grayscale
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # bluring image
    blur = cv2.GaussianBlur(gray_scale, (0, 0), sigmaX=33, sigmaY=33)

    # dividing gray_scale by blur
    divide = cv2.divide(gray_scale, blur, scale=255)

    # apply otsu threshold
    img_removed_background = cv2.threshold(
        divide, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]
    # print(img_removed_background.min())
    # print(img_removed_background.max())
    # cv2.floodFill(img_removed_background, None, (0,0),255)
    # applying morphology
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # morph = cv2.morphologyEx(img_removed_background, cv2.MORPH_CLOSE, kernel)

    # cv2.imshow("gray", gray_scale)
    # cv2.imshow("divide", divide)
    # cv2.imshow("thresh", img_removed_background)
    removed_background = (
        "./saved_images/" + img_name_ + "_removed_background.png"
    )
    cv2.imwrite(removed_background, img_removed_background)
    # cv2.imshow("morph", morph)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # background whitening
    if white_background:
        # thresh = cv2.imread(removed_background, cv2.IMREAD_UNCHANGED)
        cv2.imwrite(removed_background, img_removed_background)
        try:
            b, g, r = cv2.split(img_removed_background)

            t = [None] * 3
            u = [None] * 3
            for i, im in enumerate([b, g, r]):
                t[i], u[i] = cv2.threshold(
                    im, 255, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
                )

            img_removed_background = cv2.merge((*u,))
            cv2.imwrite(removed_background, img_removed_background)
        except ValueError:
            print("Image is not changed to white background.")

    # mask out boxes
    img_removed_background = cv2.imread(removed_background)
    img_removed_background_copy = img_removed_background.copy()

    h, w = (
        img_removed_background_copy.shape[0],
        img_removed_background_copy.shape[1],
    )

    # floodFill Seed points
    seeds = (
        (0, 0),
        (10, 10),
        (w - 1, 0),
        (w - 1, 10),
        (0, h - 1),
        (10, h - 1),
        (w - 1, h - 1),
        (w - 10, h - 10),
    )

    # Fill connected components with the given color (black).

    for seed in seeds:
        cv2.floodFill(
            img_removed_background_copy,
            None,
            seedPoint=seed,
            newVal=(0, 0, 0),
            loDiff=(5, 5, 5),
            upDiff=(5, 5, 5),
        )

    # Use Open morphological operation with threshold value - erosion then dilation. It is useful in removing noise.
    if (h + w) / 2 < 1500:
        img_removed_background_copy = cv2.morphologyEx(
            img_removed_background_copy,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)),
        )
    else:
        img_removed_background_copy = cv2.morphologyEx(
            img_removed_background_copy,
            cv2.MORPH_OPEN,  # erosion then dilation. It is useful in removing noise
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50)),
        )

    # Converting image to grayscale, and then to binary
    gray = cv2.cvtColor(img_removed_background_copy, cv2.COLOR_RGB2GRAY)
    ret, thresh_gray = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)

    # Find contours for enclosed boxes
    contours, _ = cv2.findContours(
        thresh_gray,
        cv2.RETR_EXTERNAL,  # returns only extreme outer flags. All child contours (if inside each other) are left behind.
        cv2.CHAIN_APPROX_NONE,  # all the boundary points are stored.
    )

    res1 = []
    for c in contours:
        x, y = c.T
        x = x.tolist()[0]
        y = y.tolist()[0]
        # interpolation to have smoother connected component. cubic spline interpolation
        tck, u = splprep([x, y], u=None, s=1.0, per=10)
        u_new = np.linspace(u.min(), u.max())
        x_new, y_new = splev(u_new, tck, der=0)
        res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new, y_new)]
        smoothened = np.asarray(res_array, dtype=np.int32)

        # Building a mask based on smoothened connected component. each smoothened is a contour
        mask = np.zeros_like(thresh_gray)
        # mask would change based on connected components contour smoothened (applied 255 at that position)
        cnt_ = cv2.drawContours(mask, [smoothened], -1, 255, -1)

        # Applying mask for all dimensions
        res = np.zeros_like(img_removed_background)
        res[(mask > 0)] = img_removed_background[(mask > 0)]
        res1.append(res)

    out = np.zeros_like(img_removed_background)
    for i, image in enumerate(res1):
        out += image

    path_out = "./saved_images/" + img_name_ + "_masked.png"
    # cv2.imshow('final image', out)
    cv2.imwrite(path_out, out)
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return path_out
