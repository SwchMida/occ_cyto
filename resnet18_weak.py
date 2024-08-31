import os
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix
import numpy as np
import albumentations as A
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Define Albumentations transforms for training
albumentations_transform_train = A.Compose([
    A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1),
    A.RandomResizedCrop(224, 224, scale=(0.9, 1.0), ratio=(1, 1), interpolation=cv2.INTER_LANCZOS4, p=1.0),
])

# Define transformation for testing data
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['normal', 'abnormal']
        self.file_list = self._generate_file_list()

    def _generate_file_list(self):
        file_list = []
        if 'train' in self.root_dir:
            for bag_idx in range(1, 11):
                class_idx = 0 if bag_idx <= 5 else 1  # Assign class index based on bag number
                class_folder = 'normal' if class_idx == 0 else 'abnormal'
                class_path = os.path.join(self.root_dir, f'bag_{bag_idx}')
                images = [img_name for img_name in os.listdir(class_path)]
                class_files = [(class_idx, os.path.join(class_path, img_name)) for img_name in images]
                file_list.extend(class_files)

        else:  # For test set
            class_path = os.path.join(self.root_dir, 'normal')
            images = [img_name for img_name in os.listdir(class_path)]
            class_files = [(0, os.path.join(class_path, img_name)) for img_name in images]
            file_list.extend(class_files)

            class_path = os.path.join(self.root_dir, 'abnormal')
            images = [img_name for img_name in os.listdir(class_path)]
            class_files = [(1, os.path.join(class_path, img_name)) for img_name in images]
            file_list.extend(class_files)

        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        try:
            class_idx, img_path = self.file_list[idx]
            image = Image.open(img_path).convert('RGB')

            # Apply Albumentations transforms only for training data
            if self.transform and 'train' in self.root_dir:
                image = np.array(image)
                augmented = self.transform(image=image)
                image = Image.fromarray(augmented['image'])

            # Convert the image to a PyTorch tensor
            image = transforms.ToTensor()(image)

            return image, class_idx

        except Exception as e:
            print(f"Error at index {idx}: {e}")
            raise e

# Create datasets and dataloaders
train_dataset = CustomDataset(root_dir='train_7', transform=albumentations_transform_train)
test_dataset = CustomDataset(root_dir='Test', transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load pre-trained ResNet18 model
model = torchvision.models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, len(train_dataset.classes))

# Specify device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training
num_epochs = 30
train_losses = []

# Add this line to print the length of the training dataset
print(f"Length of Training Dataset: {len(train_loader.dataset)}")


for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    average_loss = epoch_loss / len(train_loader.dataset)
    train_losses.append(average_loss)

    print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {average_loss:.16f}")

# Save the trained model after training
torch.save(model.state_dict(), 'trained_model_7.pth')


# Save training loss plot
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), train_losses, marker='o', linestyle='-', color='b')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss Over Epochs')
plt.grid(True)
plt.savefig('training_loss_plot.png')
plt.show()
plt.close()

# Calculate confusion matrix for training data
model.eval()
all_preds_train = []
all_labels_train = []

with torch.no_grad():
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds_train.extend(preds.cpu().numpy())
        all_labels_train.extend(labels.cpu().numpy())

# Save confusion matrix as an image for training data
conf_matrix_train = confusion_matrix(all_labels_train, all_preds_train)

# Save confusion matrix as an image for training data
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_train, annot=True, fmt="d", cmap="Blues", xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix (Training)")
plt.savefig('conf_matrix_train.png')
plt.close()

# Testing
model.eval()
all_preds_test = []
all_labels_test = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds_test.extend(preds.cpu().numpy())
        all_labels_test.extend(labels.cpu().numpy())

# Save confusion matrix as an image for training data
conf_matrix_test = confusion_matrix(all_labels_test, all_preds_test)        

# Save confusion matrix as an image for testing data
conf_matrix_test = confusion_matrix(all_labels_test, all_preds_test)

# Save confusion matrix as an image for testing data
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_test, annot=True, fmt="d", cmap="Blues", xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix (Testing)")
plt.savefig('conf_matrix_test.png')
plt.close()

# Print and save accuracy for training data
accuracy_train = np.trace(conf_matrix_train) / np.sum(conf_matrix_train)
print(f"Training Accuracy: {accuracy_train * 100:.2f}%")
 
# Print and save accuracy for test data
accuracy_test = np.trace(conf_matrix_test) / np.sum(conf_matrix_test)
print(f"Testing Accuracy: {accuracy_test * 100:.2f}%")