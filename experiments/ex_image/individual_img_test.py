from image_model_interface import load_image_model, extract_image_features

# Path to model weights.
model_path = r"../models/mobilenetv2_img.pth"

# Load the model once.
load_image_model(model_path=model_path)

# Test on a sample image.
# test_img =  "./test_samples/val_ha.png"
# test_img = "./test_samples/test_ang.png"
test_img = "./test_samples/train_sad.png"
label, softmax, emb = extract_image_features(test_img)

print("Predicted label:", label)
print("Softmax output:", softmax)
print("Embedding shape:", emb.shape)
