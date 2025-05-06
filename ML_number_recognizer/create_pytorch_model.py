import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import torch.nn as nn
import torch.nn.functional as F


def open_split_data_file(file):
    df = pd.read_csv(file)
    y = df.iloc[:, 0].values
    x = df.iloc[:, 1:].values.astype("float32") / 255.0
    x = x.reshape(-1, 1, 28, 28)
    return x, y


def get_model() -> torch.nn.Module:
    path = os.getcwd()
    head, tail = os.path.split(path)
    if os.path.exists("digit_classifier_tensorflow.keras"):
        model = DigitCNN()
        model.load_state_dict(torch.load("digit_classifier_pytorch.pth"))
        model.eval()
        return model
    else:
        x_train, y_train = open_split_data_file(
            os.path.join(
                head,
                "MNIST-dataset-in-different-formats-master\\" "data\\CSV format\\mnist_train.csv",
            )
        )
        model = train_model(x_train, y_train)
        torch.save(model.state_dict(), "digit_classifier_pytorch.pth")
        return model


def train_model(x, y):
    x_train_tensor = torch.tensor(x)
    y_train_tensor = torch.tensor(y).long()
    train_ds = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    model, criterion = DigitCNN(), nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 5
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")
    model.eval()
    return model


class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    path = os.getcwd()
    head, tail = os.path.split(path)

    x_test, y_test = open_split_data_file(
        os.path.join(
            head,
            "MNIST-dataset-in-different-formats-master\\" "data\\CSV format\\mnist_test.csv",
        )
    )

    x_test_tensor = torch.tensor(x_test)
    y_test_tensor = torch.tensor(y_test).long()

    test_ds = TensorDataset(x_test_tensor, y_test_tensor)

    test_loader = DataLoader(test_ds, batch_size=64, shuffle=True)

    correct, total = 0, 0
    model = get_model()
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")
