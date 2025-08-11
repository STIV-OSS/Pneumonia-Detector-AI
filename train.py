import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# ----- 1. Paths -----
data_dir = 'data'

# ----- 2. Data Augmentation & Normalization -----
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ----- 3. Datasets & DataLoaders -----
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
val_dataset   = datasets.ImageFolder(os.path.join(data_dir, 'val'),   transform=test_transform)
test_dataset  = datasets.ImageFolder(os.path.join(data_dir, 'test'),  transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False,   num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False,  num_workers=4, pin_memory=True)

# ----- 4. Model: EfficientNet -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)  # 2 classes: normal, pneumonia
model = model.to(device)

# ----- 5. Loss, Optimizer, Scheduler -----
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.3, verbose=True)

# ----- 6. Training Function -----
def evaluate(loader):
    model.eval()
    correct, total = 0, 0
    running_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            running_loss += loss.item()
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = correct / total
    avg_loss = running_loss / len(loader)
    return acc, avg_loss

# ----- 7. Training Loop -----
epochs = 12
best_val_acc = 0
patience = 4
stale = 0

for epoch in range(epochs):
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += (outputs.argmax(1) == labels).sum().item()
        train_total += labels.size(0)

    train_acc = train_correct / train_total
    train_avg_loss = train_loss / len(train_loader)
    val_acc, val_loss = evaluate(val_loader)
    scheduler.step(val_acc)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_avg_loss:.4f} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f} | Val Loss: {val_loss:.4f}")

    # Early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        stale = 0
        torch.save(model.state_dict(), "efficientnet_pneumonia_best.pth")
        print(f"Saved best model at epoch {epoch+1} (val acc {val_acc:.3f})")
    else:
        stale += 1
        if stale >= patience:
            print("Early stopping triggered.")
            break

# ----- 8. Test -----
model.load_state_dict(torch.load("efficientnet_pneumonia_best.pth"))  # Use the best
test_acc, test_loss = evaluate(test_loader)
print(f"Test Accuracy: {test_acc:.3f}, Test Loss: {test_loss:.4f}")

# ----- 9. Save Final Model -----
torch.save(model.state_dict(), "efficientnet_pneumonia_final.pth")
print("Final model saved as efficientnet_pneumonia_final.pth")
