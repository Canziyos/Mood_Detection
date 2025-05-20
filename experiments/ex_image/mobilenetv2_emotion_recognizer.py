import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image
import numpy as np
import random

class MobileNetV2EmotionRecognizer:
    # NOTE: Making data_dir optional and adding class_names means we can load for inference
    # without any image folders -just give the model path and classes.
    # If class_names is set: skip data loading, go straight to inference mode.
    # If not: do the normal training setup with data_dir and class discovery.
    # Now you can load the model for inference anywhere, anytime, with just the model weights and class names—no dataset circus required.
    # To train, just give it a data_dir and let it do its thing as before.
    # Augmentations are off for inference, so predictions are not trippy.
    # Still grayscale all the way as before.
    def __init__(self, data_dir=None, model_path="models/mobilenetv2_img.pth", batch_size=8, lr=1e-4, device=None, class_names=None):

        self.data_dir = data_dir
        self.model_path = model_path
        self.batch_size = batch_size
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if class_names is not None:
            # Inference mode: simple, no-augmentation transform
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            self.class_names = class_names
            self.num_classes = len(class_names)
        else:
            # Training mode: augmentations galore! (like before).
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomResizedCrop(224, scale=(0.9, 1.1), ratio=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            self._prepare_data()
        self._build_model()

    def _prepare_data(self):
        dataset = datasets.ImageFolder(self.data_dir, transform=self.transform)
        self.class_names = dataset.classes
        self.num_classes = len(self.class_names)

        indices = list(range(len(dataset)))
        targets = [dataset.targets[i] for i in indices]

        random.seed(42)
        random.shuffle(indices)

        train_idx, val_idx = train_test_split(
            indices,
            test_size=0.1,
            stratify=targets,
            random_state=42
        )

        self.train_loader = DataLoader(Subset(dataset, train_idx), batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(Subset(dataset, val_idx), batch_size=self.batch_size)

    def _build_model(self):
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, self.num_classes)
        for param in self.model.features.parameters():
            param.requires_grad = True
        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

    def train(self, epochs=20):
        best_acc = 0
        patience = 3
        patience_counter = 0

        for epoch in range(epochs):
            # print("started")
            self.model.train()
            total_loss = 0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            val_acc = self.evaluate()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

            self.scheduler.step()

            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        self.save_model()

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print("Classification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))

        return 100 * correct / total

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        self._build_model()
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

        
    def predict(self, image_input):
        """
        Accepts a file path (str) or a NumPy array ([H,W], [H,W,1], or [H,W,3]).
        Always converts to grayscale as expected by the model.
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("L")
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 3:
                if image_input.shape[2] == 1:
                    image_input = image_input.squeeze(-1)
                    image = Image.fromarray(image_input.astype(np.uint8)).convert("L")
                elif image_input.shape[2] == 3:
                    image = Image.fromarray(image_input.astype(np.uint8)).convert("L")
                else:
                    raise ValueError("NumPy array must have shape [H,W], [H,W,1], or [H,W,3]")
            else:
                image = Image.fromarray(image_input.astype(np.uint8)).convert("L")
        else:
            raise ValueError("Input must be a file path or a numpy array.")

        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.features(input_tensor)
            pooled = nn.AdaptiveAvgPool2d(1)(features)
            flattened = pooled.view(pooled.size(0), -1)
            logits = self.model.classifier(flattened)
            softmax_output = torch.softmax(logits, dim=1)
            predicted_index = torch.argmax(softmax_output, dim=1).item()
            predicted_label = self.class_names[predicted_index]
        return {
            "label": predicted_label,
            "softmax": softmax_output.cpu().numpy(),
            "logits": logits.cpu().numpy(),              #  added this.
            "last_hidden_layer": flattened.cpu().numpy()
        }
