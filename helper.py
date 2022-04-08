from __future__ import print_function
import cv2
import math
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
from PIL import Image
from rot_net.utils import rotate


def final_predict_digits(img, model, model_rot, rotation_corrector):
    """prediction for each digit (roi) using CNN saved model.
        Rotation corrector for each digit may be applied using rot_net saved model.

    Args:
        img (numpy ndarray): roi image for each digit

        model (keras model): model to be used for digit detection

        model_rot (keras model): model to be used for rotation detection and corrector

        rotation_corrector (bool): if True, rotation correction is performed

    Returns:
        list[int]: index of max probability for prediction
    """
    test_image = img.reshape(-1, 28, 28, 1)

    if rotation_corrector:
        # rotation corrector
        N, h, w = test_image.shape[:3]
        # size = (h, w)
        y_pred_rot = np.argmax(model_rot.predict(test_image), axis=1)
        print("Rotated degree prediction: ", y_pred_rot)
        corrected_image = rotate(test_image.reshape(28, 28), -y_pred_rot)
        resized_corrected_img = Image.fromarray(corrected_image)
        resized_corrected_img = np.array(
            resized_corrected_img.resize((28, 28), Image.ANTIALIAS)
        )
        test_image = resized_corrected_img.reshape(-1, 28, 28, 1)
        # plt.imshow(resized_corrected_img, cmap="gray")
        # plt.show()
    print("Probabilities: ", model.predict(test_image))

    return np.argmax(model.predict(test_image))


def attach_label(img_name, first_original_img, label, x, y):
    """attach label to original image for saving and presentation

    Args:
        img_name (str): name of the test image
        first_original_img (numpy ndarray): original test image
        label (int): label of the digit
        x (float): center x coordinate of the digit
        y (float): center y coordinate of the digit

    Returns:
        str: output path of the test labeled image
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    box_x = int(x) - 70
    box_y = int(y) + 50

    cv2.rectangle(
        first_original_img,
        (box_x, box_y + 5),
        (box_x + 50, box_y - 50),
        (0, 255, 0),
        -1,
    )
    cv2.putText(
        first_original_img,
        str(label),
        (box_x, box_y),
        font,
        2,
        (255, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        first_original_img,
        "x:" + str(round(x, 1)),
        (int(x), int(y) - 60),
        font,
        0.75,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        first_original_img,
        "y:" + str(round(y, 1)),
        (int(x), int(y) - 40),
        font,
        0.75,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )

    name_label = img_name + "_labeled.jpg"
    output_path = "./saved_images/" + name_label
    cv2.imwrite(output_path, first_original_img)
    # if you wish not stop for each roi then comment below out
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return output_path


def roi_processor(region_of_interest):
    """processing roi for each digit have output shape (28, 28)

    Args:
        region_of_interest (numpy ndarray): region of interest for each digit

    Returns:
        numpy ndarray: processed roi
    """
    original_size = 22
    image_size = 28
    region_of_interest = cv2.bitwise_not(region_of_interest)
    row, col = region_of_interest.shape

    if row > col:  # 128, 88
        fact = original_size / row  # 0.17
        row = original_size  # 22
        col = int(round(col * fact))  # 15
    else:
        fact = original_size / col
        col = original_size
        row = int(round(row * fact))
    region_of_interest = cv2.resize(region_of_interest, (col, row))

    # padding
    cols_padding = (
        int(math.ceil((image_size - col) / 2.0)),  # 7
        int(math.floor((image_size - col) / 2.0)),  # 6
    )
    rows_padding = (
        int(math.ceil((image_size - row) / 2.0)),  # 3
        int(math.floor((image_size - row) / 2.0)),  # 3
    )

    # applying padding
    region_of_interest = np.lib.pad(
        region_of_interest, (rows_padding, cols_padding), "constant"
    )
    return region_of_interest
