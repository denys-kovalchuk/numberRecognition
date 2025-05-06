from PIL import Image
import numpy as np
import torch
import os
from create_pytorch_model import get_model


def get_image(name):
    img = Image.open(name).convert("L")
    img = img.resize((28, 28))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)
    return img_array


def main(file):
    model = get_model()
    path = os.getcwd()
    head, tail = os.path.split(path)

    img = get_image(os.path.join(head, file))
    with torch.no_grad():
        output = model(img)
        prediction = torch.argmax(output, dim=1).item()
    print(prediction)
    return prediction


if __name__ == "__main__":
    file = "4p.png"
    main(file)
