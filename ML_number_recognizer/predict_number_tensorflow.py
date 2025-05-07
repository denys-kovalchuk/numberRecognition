from ML_number_recognizer.create_tensorflow_model import get_model
from PIL import Image
import numpy as np
import os


def get_image(name):
    img = Image.open(name).convert("L")
    img = img.resize((28, 28))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array


def main(file):
    model = get_model()
    path = os.getcwd()
    head, tail = os.path.split(path)

    img = get_image(os.path.join(head, file))
    predict = model.predict(img)
    print(predict.argmax())
    return predict.argmax()


if __name__ == "__main__":
    file = "4p.png"
    main(file)
