import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from PIL import Image
import pandas as pd
import os
import sys
import argparse
import datetime
import hashlib

MODEL_PATH = "efficientnet_pneumonia_best.pth"
FEEDBACK_CSV = "feedback_log.csv"
FEEDBACK_IMAGES = "feedback_images"
AUDIT_LOG = "retraining_audit_log.csv"
EPOCHS = 5

# --- CLI args for feedback oversample weight ---
parser = argparse.ArgumentParser()
parser.add_argument("--weight", type=int, default=10, help="Feedback oversample weight (default=10)")
args = parser.parse_args()
FEEDBACK_OVERSAMPLE = args.weight

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Original training data (update these paths!) ---
train_dir = "chest_xray/train"   # <--- UPDATE as needed!
test_dir = "chest_xray/test"     # <--- UPDATE as needed!

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
test_ds = datasets.ImageFolder(test_dir, transform=test_transform)

# --- 2. Feedback Dataset ---
class FeedbackDataset(Dataset):
    def __init__(self, csv_path, images_dir, transform):
        self.data = pd.read_csv(csv_path, header=None)
        self.images_dir = images_dir
        self.transform = transform
        self.label_map = {"Normal": 0, "Pneumonia": 1}
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.data.iloc[idx,0])
        image = Image.open(img_path).convert('L')
        image = self.transform(image)
        label = self.label_map[self.data.iloc[idx,3]]
        return image, label

if os.path.exists(FEEDBACK_CSV):
    feedback_ds = FeedbackDataset(FEEDBACK_CSV, FEEDBACK_IMAGES, train_transform)
    dataset = ConcatDataset([train_ds] + [feedback_ds]*FEEDBACK_OVERSAMPLE)
    print(f"Oversampling feedback {FEEDBACK_OVERSAMPLE}x, total training size: {len(dataset)}")
else:
    dataset = train_ds
    print("No feedback samples found. Training only on original data.")

loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

# --- 3. Model setup (EfficientNet-B0) ---
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)

# --- 4. Training ---
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(loader):.4f} | Accuracy: {correct/total:.3f}")

torch.save(model.state_dict(), MODEL_PATH)
print("Model updated and saved!")

# --- 5. Test evaluation ---
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)
model.eval()
test_correct, test_total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(1)
        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)
test_acc = test_correct / test_total
print(f"Test accuracy after retrain: {test_acc:.3f}")

# --- 6. Audit log ---
model_bytes = open(MODEL_PATH, "rb").read()
model_hash = hashlib.sha256(model_bytes).hexdigest()
log_line = f"{datetime.datetime.now()},{model_hash},{len(train_ds)},{len(feedback_ds) if os.path.exists(FEEDBACK_CSV) else 0},{test_acc:.4f}\n"
with open(AUDIT_LOG, "a") as logf:
    logf.write(log_line)
