import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import os
from PIL import Image
import numpy as np


def open_split_data_file(file):
    df = pd.read_csv(file)
    y = df.iloc[:, 0].values
    x = df.iloc[:, 1:].values.astype("float32")
    x /= 255.0
    x = x.reshape(-1, 28, 28, 1)
    return x, y


def get_model():
    path = os.getcwd()
    head, tail = os.path.split(path)
    if os.path.exists("digit_classifier_tensorflow.keras"):
        return tf.keras.models.load_model("digit_classifier_tensorflow.keras")
    else:
        x_train, y_train = open_split_data_file(
            os.path.join(
                head,
                "MNIST-dataset-in-different-formats-master\\" "data\\CSV format\\mnist_train.csv",
            )
        )
        model = train_model(x_train, y_train)
        model.save("digit_classifier_tensorflow.keras")
        return model


def train_model(x, y):
    model = models.Sequential(
        [
            layers.Reshape((28, 28, 1), input_shape=(28, 28)),  # Reshape for CNN input
            layers.Conv2D(32, (3, 3), activation="relu"),  # Convolutional layer
            layers.MaxPooling2D((2, 2)),  # Pooling layer
            layers.Conv2D(64, (3, 3), activation="relu"),  # Another convolutional layer
            layers.MaxPooling2D((2, 2)),  # Pooling layer
            layers.Flatten(),  # Flatten the data to feed into the fully connected layer
            layers.Dense(64, activation="relu"),  # Fully connected layer
            layers.Dense(10, activation="softmax"),  # Output layer for 10 digits
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x, y, epochs=5)
    return model


if __name__ == "__main__":

    model = get_model()
    path = os.getcwd()
    head, tail = os.path.split(path)
    x_test, y_test = open_split_data_file(
        os.path.join(
            head,
            "MNIST-dataset-in-different-formats-master\\" "data\\CSV format\\mnist_test.csv",
        )
    )
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc}")

    def get_image(name):
        img = Image.open(name).convert("L")
        img = img.resize((28, 28))
        img_array = np.array(img).astype("float32") / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)
        return img_array

    img = get_image(os.path.join(head, "7.png"))
    predict_img = model.predict(img)
    print(predict_img)
    print(predict_img.argmax())
