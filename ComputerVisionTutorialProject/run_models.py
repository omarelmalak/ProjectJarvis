import tensorflow as tf
import cv2
import numpy as np

if __name__ == "__main__":
    model = tf.keras.models.load_model("./trained_models/vgg16_5epoch.ckpt")

    image = cv2.imread("./media/input_media/jilly_mirror_image.png")
    image = cv2.resize(image, (64, 64))
    cv2.imshow('image sample', image)
    cv2.waitKey(0)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    numpy_image = np.array(image_rgb)
    numpy_image = np.expand_dims(numpy_image, axis=0)

    predictions = model.predict(numpy_image)

    print(predictions)

    predicted_class = np.argmax(predictions, axis=1)