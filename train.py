from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping


target_size = (256, 256)

# Define the CNN model
model = Sequential()

# Add a convolutional layer with 32 filters, a 3x3 kernel, and specify the input shape
model.add(Conv2D(32, (3, 3), input_shape=(256, 256, 3)))
model.add(Activation('relu')) # Activation function to add non-linearity
model.add(MaxPooling2D(pool_size=(2, 2))) # Pooling layer to reduce dimensions

# Add another convolutional layer
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add a third convolutional layer
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output of the convolutional layers to feed into a dense layer
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5)) # Dropout to prevent overfitting
model.add(Dense(3)) # Final layer with 3 neurons for each class
model.add(Activation('softmax')) # Softmax for multi-class classification

model.summary() # Print a summary of the model

# Compile the model with categorical_crossentropy loss for multi-class classification,
# the adam optimizer, and track accuracy of the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# ImageDataGenerators for loading and augmenting images
train_images = 764
validation_images = 191
batch_size = 16

steps_per_epoch = train_images // batch_size  # Use floor division to ensure an integer result
validation_steps = validation_images // batch_size  # Likewise for validation

print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")
# Training data generator with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255, # Rescale pixel values for normalization
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Validation data generator without augmentation, only rescaling
validation_datagen = ImageDataGenerator(rescale=1./255)



# Setup the training data generator to fetch images from the directory
train_generator = train_datagen.flow_from_directory(
    'dataset/train', # Replace with path to your train directory
    target_size=target_size, # Target size for the images
    batch_size=batch_size,
    class_mode='categorical') # Classes are categorical

# Assuming 'train_generator' is the same generator you used for training
class_labels = list(train_generator.class_indices.keys())

print(class_labels)

# Setup the validation data generator to fetch images from the directory
validation_generator = validation_datagen.flow_from_directory(
    'dataset/validation', # Replace with path to your validation directory
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical')

# Set up early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

# Fit the model with the early stopping callback
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=50,  # Set a large enough number to allow early stopping to kick in
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[early_stopping]  # Include the early stopping callback here
)

model.save('model.h5') # Save the model to a file