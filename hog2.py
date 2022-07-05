import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure
import cv2

if __name__ == '__main__':

    image = cv2.imread('nin.png')
    img = cv2.imread('nin.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, channel_axis=-1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(img)
    ax1.set_title('სურათი')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('ორიენტირებული გრადიენტის ჰისტოგრამა')
    plt.show()