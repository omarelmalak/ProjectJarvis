import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path_to_image_data = "raw_data/image_data.npy"
path_to_metadata = "raw_data/metadata.csv"
images_base_address = "images/"


class Helpers:
    def extract_to_images(num_images: int, images, metadata) -> None:
        for i in range(num_images):
            image = images[i]

            reshaped_image = image.reshape((64, 64, 3))

            # DISPLAY IMAGE:
            """
            plt.imshow(reshaped_image)
            plt.axis("off")
            plt.show()
            """

            file_name = "radio.png"

            plt.imsave(images_base_address + file_name, reshaped_image)


if __name__ == "__main__":
    images = np.load(path_to_image_data)

    metadata = pd.read_csv(path_to_metadata)

    # print(len(images))
    # print(metadata.head())

    Helpers.extract_to_images(5, images, metadata)
