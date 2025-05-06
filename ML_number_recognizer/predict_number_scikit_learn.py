from ML_number_recognizer.create_scikit_learn_model import get_model
import os
from PIL import Image
import numpy as np


def get_image(name, scaler):
    img = Image.open(name).convert("L")
    img = img.resize((28, 28))
    img_array = np.array(img).reshape(1, -1)
    return scaler.transform(img_array)


def main(file: str) -> int:
    model, scaler = get_model()
    path = os.getcwd()
    head, tail = os.path.split(path)
    print(path, head)
    img = get_image(os.path.join(head, file), scaler)
    # img = get_image(os.path.join(path, file.filename), scaler)
    predict_img = model.predict(img)
    print(predict_img[0])
    return predict_img[0]


if __name__ == "__main__":
    file = "3.png"
    main(file)
