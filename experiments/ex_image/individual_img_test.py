from image_model_interface import load_image_model, extract_image_features

# Path to model weights.
model_path = r"../models/mobilenetv2_emotion.pth"

# Load the model once.
load_image_model(model_path=model_path)

# Test on a sample image.
# test_img =  r"../Dataset/Images/val/val_ha.png"
# test_img = r"../Dataset/Images/test/test_ang.png"
test_img = r"../Dataset/Images/Train/train_sad.png"
label, softmax, emb = extract_image_features(test_img)

print("Predicted label:", label)
print("Softmax output:", softmax)
print("Embedding shape:", emb.shape)
