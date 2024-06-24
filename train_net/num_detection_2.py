import torch
import torchvision
from tqdm import tqdm
import matplotlib

matplotlib.use('TkAgg')  # 或者 'Qt5Agg', 'GTK3Agg' 等
import matplotlib.pyplot as plt
from torchvision import transforms

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(torch.cuda.is_available())  # 检查 CUDA 是否可用

# 将数据集的图片转为张量，并归一化
transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load datasets
path = '../data/'
train_data = torchvision.datasets.MNIST(root=path, train=True, transform=transform, download=True)
value_data = torchvision.datasets.MNIST(root=path, train=False, transform=transform)

# Data loaders
BATCH_SIZE = 256
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
value_loader = torch.utils.data.DataLoader(dataset=value_data, batch_size=BATCH_SIZE)

num_classes = len(value_data.classes)


# print(f"The value_loader has {num_classes} classes.")

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # 定义各个层，而不是直接放在Sequential中
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.lrelu1 = torch.nn.LeakyReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2)

        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.lrelu2 = torch.nn.LeakyReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(256)
        self.lrelu3 = torch.nn.LeakyReLU()
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2)

        self.flatten = torch.nn.Flatten()

        self.fc1 = torch.nn.Linear(256 * 6 * 6, 1024)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.lrelu4 = torch.nn.LeakyReLU()

        self.fc2 = torch.nn.Linear(1024, 256)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.lrelu5 = torch.nn.LeakyReLU()

        self.fc3 = torch.nn.Linear(256, num_classes)

    def forward(self, x):
        # torch.Size([256, 32, 14, 14])
        x = self.pool1(self.lrelu1(self.bn1(self.conv1(x))))
        # torch.Size([256, 64, 5, 5])
        x = self.pool2(self.lrelu2(self.bn2(self.conv2(x))))
        # torch.Size([256, 128, 2, 2])
        x = self.pool3(self.lrelu3(self.bn3(self.conv3(x))))

        x = self.flatten(x)
        x = self.dropout1(self.lrelu4(self.fc1(x)))
        x = self.dropout2(self.lrelu5(self.fc2(x)))

        x = self.fc3(x)
        return x


# Initialize model, loss function, optimizer, and learning rate scheduler
net = Net().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Training loop
EPOCHS = 10
history = {'value Loss': [], 'value Accuracy': []}

for epoch in range(1, EPOCHS + 1):
    net.train()
    process_bar = tqdm(train_loader, unit='step')
    for step, (images, labels) in enumerate(process_bar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        predictions = torch.argmax(outputs, dim=1)
        accuracy = torch.sum(predictions == labels).item() / labels.size(0)
        process_bar.set_description(f"[{epoch}/{EPOCHS}] Loss: {loss.item():.4f}, Acc: {accuracy:.4f}")

    scheduler.step()

    # Evaluation
    net.eval()
    correct, total_loss = 0, 0
    with torch.no_grad():
        for images, labels in value_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            predictions = torch.argmax(outputs, dim=1)
            total_loss += loss.item()
            correct += torch.sum(predictions == labels).item()

    value_accuracy = correct / len(value_data)
    value_loss = total_loss / len(value_loader)
    history['value Loss'].append(value_loss)
    history['value Accuracy'].append(value_accuracy)
    print(f"value Loss: {value_loss:.4f}, value Accuracy: {value_accuracy:.4f}")

# Plot the value loss and accuracy
plt.plot(history['value Loss'], label='value Loss')
plt.legend(loc='best')
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

plt.plot(history['value Accuracy'], color='red', label='value Accuracy')
plt.legend(loc='best')
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# Save the model
torch.save(net.state_dict(), '../model/model_2_2.pth')
