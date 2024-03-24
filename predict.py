import numpy as np
from keras.models import load_model
from keras.preprocessing import image

model = load_model('model.h5')

# Load an image file, resizing it to 256x256 pixels (as the model expects)
img = image.load_img('/Users/ollej/Dev/kth/kex/dataset/train/ice-free/SYFW0584.JPG', target_size=(256, 256))

# Convert the image to a numpy array
img_array = image.img_to_array(img)

# Add a fourth dimension to the image (since Keras expects a list of images, not a single image)
img_array = np.expand_dims(img_array, axis=0)

# Normalize the image
img_array /= 255.0

# Use the model to make a prediction
predictions = model.predict(img_array)

# The output `predictions` will be an array containing probabilities for each class
# If you want the class with the highest probability:
predicted_class = np.argmax(predictions)

print(predicted_class)
