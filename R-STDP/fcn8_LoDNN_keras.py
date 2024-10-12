import os
import tensorflow as tf
import tensorflow_addons as tfa
import cv2
from glob import glob
from datetime import datetime
import numpy as np
import time

class LoDNN():
    def __init__(self):
        # Suppress TF warnings
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

        # Get reproducable results
        tf.keras.utils.set_random_seed(72)
        tf.config.experimental.enable_op_determinism()
        tf.random.set_seed(72)

        self.hparams = {
            "input_shape": (400, 200, 1),
        }
        self.model = self.create_LoDNN_model(input_shape=self.hparams["input_shape"])
        self.model.load_weights('./data/fcnData/newDataset_model_256epoch.h5')

    def create_LoDNN_model(self, input_shape):
        inp = x = tf.keras.Input(input_shape)
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same")(x)
        x = tf.keras.layers.LeakyReLU()(x)

        x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same")(x)
        x = tf.keras.layers.LeakyReLU()(x)

        x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)

        x = tf.keras.layers.Conv2D(
            filters=128, kernel_size=3, strides=1, padding="same", dilation_rate=1
        )(x)
        x = tf.keras.layers.LeakyReLU()(x)

        x = tf.keras.layers.Conv2D(
            filters=128, kernel_size=3, strides=1, padding="same", dilation_rate=1
        )(x)
        x = tf.keras.layers.LeakyReLU()(x)

        x = tf.keras.layers.Conv2D(
            filters=128, kernel_size=3, strides=1, padding="same", dilation_rate=1
        )(x)
        x = tf.keras.layers.LeakyReLU()(x)

        x = tf.keras.layers.Conv2D(
            filters=128, kernel_size=3, strides=1, padding="same", dilation_rate=1
        )(x)
        x = tf.keras.layers.LeakyReLU()(x)

        x = tf.keras.layers.Conv2D(
            filters=128, kernel_size=3, strides=1, padding="same", dilation_rate=1
        )(x)
        x = tf.keras.layers.LeakyReLU()(x)

        x = tf.keras.layers.Conv2D(
            filters=128, kernel_size=3, strides=1, padding="same", dilation_rate=1
        )(x)
        x = tf.keras.layers.LeakyReLU()(x)

        x = tf.keras.layers.Conv2D(
            filters=128, kernel_size=3, strides=1, padding="same", dilation_rate=1
        )(x)
        x = tf.keras.layers.LeakyReLU()(x)

        x = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding="same")(x)

        x = tf.keras.layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(x)


        x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same")(x)
        x = tf.keras.layers.LeakyReLU()(x)

        x = tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, padding="same")(x)
        x = tf.keras.layers.LeakyReLU()(x)

        x = tf.keras.layers.Activation("sigmoid")(x)

        model = tf.keras.Model(inp, x)
        return model

    def generate_inference(self, dataset, threshold=0.5):
        predictions = self.model.predict(dataset, verbose=0)
        segmentation = predictions > threshold
        mask = np.dot(segmentation, 190)
        mask = np.pad(mask, ((0, 0), (0, 0), (0, 0), (1, 1)))

        result = np.clip(mask, 0, 255).astype(np.uint8)
        result = result[:,:,:,1].reshape(50,25)
        return result

    def generateNewState(self, test_img,state_shape=(8, 4)):
        cropped = cv2.resize(test_img,(200,400))
        cropped = cropped.reshape(1,400,200,1)
        result = self.generate_inference(cropped)
        result = np.float32(np.absolute(result))
        img_shape = result.shape
        cropped_img = result[30:img_shape[0], 6:img_shape[1] - 6]
        img_buf = cv2.rotate(cropped_img, cv2.ROTATE_90_CLOCKWISE)
        new_state = cv2.resize(img_buf, (state_shape[1], state_shape[0]), interpolation=cv2.INTER_AREA)
        new_state = np.where(new_state > 32, 1, 0)
        return new_state

    def printInferenceTime(self):
        begin = time.time()
        state = self.generateNewState(cv2.imread('../lidar_data/croppedData/velodyne_bv_road/um_000014.png',cv2.IMREAD_GRAYSCALE))
        end = time.time()
        print(end-begin)
        #print(state)
