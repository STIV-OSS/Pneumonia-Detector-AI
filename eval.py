import torch
from torchvision import models, transforms, datasets
import torch.nn as nn
from torch.utils.data import DataLoader
import os

data_dir = 'data'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dataset  = datasets.ImageFolder(os.path.join(data_dir, 'test'),  transform=test_transform)

test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False,  num_workers=4, pin_memory=True)

# --- Define model architecture matching training ---
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load("efficientnet_pneumonia_best.pth", map_location=device))
model.eval()
model.to(device)


import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# --- Get all test predictions and probabilities ---
all_labels = []
all_preds = []
all_probs = []

model.eval()
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # probability for class 1 (PNEUMONIA)
        pred = logits.argmax(1).cpu().numpy()
        all_labels.extend(y.cpu().numpy())
        all_preds.extend(pred)
        all_probs.extend(probs)

# --- Confusion Matrix ---
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:\n", cm)

# --- Classification Report (precision, recall, f1) ---
print("\nClassification Report:\n", classification_report(all_labels, all_preds, target_names=['Normal', 'Pneumonia']))

# --- ROC Curve & AUC ---
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
auc_score = roc_auc_score(all_labels, all_probs)
print(f"ROC AUC: {auc_score:.3f}")

plt.figure(figsize=(5,5))
plt.plot(fpr, tpr, label=f'AUC={auc_score:.2f}')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.show()
