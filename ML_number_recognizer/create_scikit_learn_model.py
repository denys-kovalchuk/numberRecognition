import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd
import joblib
from PIL import Image
import os


def open_split_data_file(file, scaler, train=True):
    df = pd.read_csv(file)
    y = df.iloc[:, 0]
    x = df.iloc[:, 1:]
    if train:
        x_scaled = scaler.fit_transform(x.values)
    else:
        x_scaled = scaler.transform(x.values)
    return x_scaled, y


def train_model(x, y):
    model = SVC(kernel="rbf", gamma=0.001)
    model.fit(x, y)
    joblib.dump(model, "digit_classifier_scikit.joblib")
    return model


def get_model():
    path = os.getcwd()
    head, tail = os.path.split(path)
    scaler = StandardScaler()
    print(path, head)
    if os.path.exists("ML_number_recognizer\\digit_classifier_scikit.joblib") and os.path.exists(
        "ML_number_recognizer\\scaler_scikit_learn.joblib"
    ):
        return joblib.load("ML_number_recognizer\\digit_classifier_scikit.joblib"), joblib.load(
            "ML_number_recognizer\\scaler_scikit_learn.joblib"
        )
    else:
        x_train, y_train = open_split_data_file(
            os.path.join(
                head,
                "MNIST-dataset-in-different-formats-master\\" "data\\CSV format\\mnist_train.csv",
            ),
            scaler,
        )
        model = train_model(x_train, y_train)
        joblib.dump(scaler, "scaler_scikit_learn.joblib")
        return model, scaler


def get_image(name, scaler):
    img = Image.open(name).convert("L")
    img = img.resize((28, 28))
    img_array = np.array(img).reshape(1, -1)
    return scaler.transform(img_array)


if __name__ == "__main__":
    model, scaler = get_model()
    path = os.getcwd()
    head, tail = os.path.split(path)
    # X_test, y_test = open_split_data_file(os.path.join(head, 'MNIST-dataset-in-different-formats-master\\'
    #                                                          'data\\CSV format\\mnist_test.csv'), scaler,
    #                                       train=False)
    # predict = model.predict(X_test)

    img = get_image(os.path.join(head, "7.png"), scaler)
    predict_img = model.predict(img)
    # print("Accuracy:", accuracy_score(y_test, predict))
    # print("\nClassification Report:\n", classification_report(y_test, predict))
    # print(predict)
    print(predict_img)
