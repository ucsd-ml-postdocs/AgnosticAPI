from tensorflow.keras.applications import MobileNetV2

# Define the model name (e.g., VGG16, ResNet50, MobileNetV2)
model_name = "MobileNetV2"

# Include the top (fully connected) layer for classification (default: False)
include_top = True

# Set weights ('imagenet' for pre-trained weights)
weights = 'imagenet'

# Download and create the model
model = MobileNetV2(weights=weights, include_top=include_top)

# Save the model (adjust path as needed)
model.save('MobileNetv2_model.keras')

# Print a confirmation message
print(f"Downloaded and saved {model_name} pre-trained model")
