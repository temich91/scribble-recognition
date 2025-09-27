import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms.v2 as tfs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS_PATH = "models"
to_tensor_tf = tfs.Compose([tfs.ToImage(), tfs.ToDtype(torch.float32, scale=True)])

# download MNIST dataset
train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=to_tensor_tf
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=to_tensor_tf
)

BATCH_SIZE = 16

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1) # (1, 28, 28) -> (16, 24, 24)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # (16, 24, 24) -> (16, 12, 12)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1) # (16, 12, 12) -> (32, 10, 10)
        # pool2 (32, 10, 10) -> (32, 5, 5)
        self.fc1 = nn.Linear(32 * 5 * 5, 130) # (800, 130)
        self.fc2 = nn.Linear(130, 10)
    
    def forward(self, x):
        x = self.pool(f.relu(self.conv1(x))) # (1, 28, 28) -> (16, 24, 24) -> (16, 12, 12)
        x = self.pool(f.relu(self.conv2(x))) # (16, 12, 12) -> (32, 10, 10) -> (32, 5, 5)
        x = x.flatten(1) # (32, 5, 5) -> (32, 25)
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# training parameters
LEARNING_RATE = 0.01
MOMENTUM = 0.9
N_EPOCHS = 2

def train_model(model, train_data, verbose=True):
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    for epoch in range(N_EPOCHS):
        epoch_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if verbose:
            print(f"epoch: {epoch + 1}, loss: {epoch_loss / len(train_loader)}")
    torch.save(net.state_dict(), f"{MODELS_PATH}/cnn.pt")


net = ConvNet()
train_model(net, train_data)


# model = ConvNet()
# model.load_state_dict(torch.load(f"{MODELS_PATH}/cnn.pt", weights_only=True))

def test(model: ConvNet, test_data):
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, true = data
            inputs = inputs.to(device)
            true = true.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, dim=1)
            correct += (pred == true).sum().item()
            total += len(true)
    print(f"Accuracy: {(correct / total * 100):.2f}") # 98.74 (120, 10) linear2

def predict(data):
    model = ConvNet()
    model.eval()
    output = model(data)
    _, prediction = torch.max(output, dim=1)
    return prediction
