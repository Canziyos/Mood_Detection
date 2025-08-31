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
    def __init__(self, data_dir, model_path="models/mobilenetv2_emotion.pth", batch_size=32, lr=3e-4, device=None):
        self.data_dir = data_dir
        self.model_path = model_path
        self.batch_size = batch_size
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # === Data Augmentation for Robustness ===
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize to a slightly larger size.
            transforms.RandomCrop(224),     # Then crop to 224x224 as expected by MobileNetV2.
            transforms.RandomHorizontalFlip(p=0.5),  # Adds variability, especially for facial symmetry
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),  # Color variability
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standard ImageNet normalization for RGB.
                                 std=[0.229, 0.224, 0.225])
        ])

        self.model = None
        self.class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
        self.num_classes = len(self.class_names)

         # Only prepare data if training is needed.
        if self.data_dir and self.data_dir != "dummy":
         self._prepare_data()

        # Build and load model
        self._build_model()

    def _prepare_data(self):
        # Load dataset and apply transformations.
        dataset = datasets.ImageFolder(self.data_dir, transform=self.transform)
        self.class_names = dataset.classes
        self.num_classes = len(self.class_names)

        # Stratified split ensures class balance
        indices = list(range(len(dataset)))
        targets = [dataset.targets[i] for i in indices]

        train_idx, val_idx = train_test_split(
            indices, test_size=0.1, stratify=targets, random_state=42
        )

        self.train_loader = DataLoader(Subset(dataset, train_idx), batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(Subset(dataset, val_idx), batch_size=self.batch_size, num_workers=4)

    def _build_model(self):
        # Load pretrained MobileNetV2 with ImageNet weights
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        # Adjust the final classifier to match number of classes in our dataset
        self.model.classifier[1] = nn.Linear(self.model.last_channel, self.num_classes)

        # Enable full fine-tuning
        for param in self.model.features.parameters():
            param.requires_grad = True

        self.model = self.model.to(self.device)

        # Use label smoothing to improve generalization
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # AdamW is preferred over Adam for regularization
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        # Cosine learning rate schedule helps avoid getting stuck in bad local minima.
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10)

    def train(self, epochs=25):
        best_acc = 0
        patience = 5
        patience_counter = 0

        for epoch in range(epochs):
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

            # Early stopping based on validation accuracy.
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                self.save_model()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

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

        # Display standard classification metrics.
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
        Accepts either:
        - a file path to an image, or
        - a NumPy array (RGB or grayscale).
        Automatically handles both grayscale and RGB inputs.
        """
        from PIL import Image

        # Load and prepare the image.
        if isinstance(image_input, str):
            image = Image.open(image_input)
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                image = Image.fromarray(image_input.astype(np.uint8), mode='L')
            elif image_input.ndim == 3:
                if image_input.shape[2] == 1:
                    image = Image.fromarray(image_input.squeeze(-1).astype(np.uint8), mode='L')
                elif image_input.shape[2] == 3:
                    image = Image.fromarray(image_input.astype(np.uint8), mode='RGB')
                else:
                    raise ValueError("Unsupported channel format in numpy array.")
            else:
                raise ValueError("Unsupported numpy array shape.")
        else:
            raise ValueError("Input must be a file path or a numpy array.")

        if image.mode != "RGB":
            image = image.convert("RGB")

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
            "last_hidden_layer": flattened.cpu().numpy()
        }
