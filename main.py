import os
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
from preprocess import preprocess_images_masking
from helper import final_predict_digits, attach_label, roi_processor
from rot_net.utils import angle_error


def get_output_image(
    img_name,
    model,
    model_rot,
    rotation_corrector=False,
    white_background=False,
):
    """main function to detect digits in image

    Args:
        img_name (str): name of the test image

        model (keras model): model to be used for digit detection

        model_rot (keras model): model to be used for rotation detection and corrector

        rotation_corrector (bool): if True, rotation correction is performed

        white_background (bool): if True, background whitening will be applied for image

    Returns:
        str: path of the saved labeled output
    """

    img_name_ = img_name.replace(".png", "")
    path_images = "./test_images/"
    path_original = path_images + img_name

    path_out = preprocess_images_masking(
        path_original, img_name_, white_background
    )
    img = cv2.imread(path_out, cv2.IMREAD_GRAYSCALE)  # masked image grayscale
    # img_org = cv2.imread(path_out)  # masked image original
    img_first = cv2.imread(path_original)  # original image

    # kernel = np.ones((5, 5), np.uint8)
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # put threshold on image
    # re, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    # contours, hierarchy = cv2.findContours(
    #     thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    # )
    contours, hierarchy = cv2.findContours(
        img,
        cv2.RETR_CCOMP,  # retrieves all of the contours and organizes them into a two-level hierarchy. At the top level, there are external boundaries of the components. At the second level, there are boundaries of the holes. If there is another contour inside a hole of a connected component, it is still put at the top level.
        cv2.CHAIN_APPROX_SIMPLE,  # or CHAIN_APPROX_NONE
    )

    for i, count in enumerate(contours):
        x, y, w, h = cv2.boundingRect(
            count
        )  # get approximate rectangle around each contour

        # hierarchy should have a parent
        if hierarchy[0][i][3] != -1 and w > 10 and h > 17:
            # put rectangle for each digit
            # cv2.rectangle(img_org, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(img_first, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # crop image
            region_of_interest = img[y : y + h, x : x + w]

            region_of_interest = roi_processor(region_of_interest)

            # add dilate and erode to have smoother image (closing) - remove small black dots inside digits
            region_of_interest = cv2.dilate(region_of_interest.copy(), None)
            region_of_interest = cv2.erode(region_of_interest.copy(), None)

            cv2.imshow("region_of_interest", region_of_interest)
            # thre, fn = cv2.threshold(
            #     region_of_interest, 127, 255, cv2.THRESH_BINARY
            # )

            # predict digit
            pred = final_predict_digits(
                region_of_interest,
                model,
                model_rot,
                rotation_corrector,
            )
            print("Predicted digit: ", pred)

            # put label for each image. a circle which completely covers the object with minimum area. x,y are the center of the circle
            (x, y), radius = cv2.minEnclosingCircle(count)

            output_path_labeled = attach_label(img_name_, img_first, pred, x, y)
            print(f"x:{x}, y:{y}")
            print("*****" * 20)

    return output_path_labeled


if __name__ == "__main__":

    # TODO: Add feature to train at the same time with CNN and rot_net
    # loading pre trained models from CNN (digit_classifier3.h5 or digit_classifier4.h5) and rot_net
    model = load_model(
        "./convolutional_net/saved_models/digit_classifier3.h5", compile=False
    )
    model_rot = load_model(
        "./rot_net/saved_models/rotation_finder.h5",
        custom_objects={"angle_error": angle_error},
    )

    test_images_names = list(os.listdir(r"./test_images/"))

    for test_img_name in test_images_names:
        print("test image: ", test_img_name)
        output_path_labeled = get_output_image(
            test_img_name,
            model,
            model_rot,
            # TODO: need more investigation for rotation_corrector is needed
            rotation_corrector=True,
            white_background=True,
        )
        image = plt.imread(output_path_labeled)
        plt.imshow(image)
        plt.title(f"Final Labeled Image {test_img_name}")
        plt.show()
