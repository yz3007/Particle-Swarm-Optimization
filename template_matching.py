"""
template matching
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html
opencv
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


def single_shot(img_rgb, img_gray, template, height, width):
    res = cv2.matchTemplate(img_gray, template, cv2.TM_SQDIFF)

    print(res.shape)
    plt.imshow(res, cmap='gray')
    plt.show()

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    print('min_val:', min_val)
    print('min_loc:', min_loc)

    top_left = min_loc  ##Change to max_loc for all except for TM_SQDIFF
    bottom_right = (top_left[0] + width, top_left[1] + height)
    cv2.rectangle(img_rgb, top_left, bottom_right, (255, 0, 0), 2)

    cv2.imshow("Matched image", img_rgb)
    cv2.waitKey()
    cv2.destroyAllWindows()


def multiple_shots(img_rgb, img_gray, template, height, width):
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

    threshold = 0.5  # For TM_CCOEFF_NORMED, larger values means good fit
    loc = np.where(res >= threshold)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + width, pt[1] + height), (255, 0, 0), 1)

    cv2.imshow("Matched image", img_rgb)
    cv2.waitKey()
    cv2.destroyAllWindows()


def main():
    img_rgb = cv2.imread('SourceIMG.jpeg')
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('TemplateIMG.jpeg', 0)

    height, width = template.shape[::]

    single_shot(img_rgb, img_gray, template, height, width)
    multiple_shots(img_rgb, img_gray, template, height, width)


if __name__ == '__main__':
    main()
