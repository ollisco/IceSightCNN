import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report

# Load the trained model
model = load_model('model.h5')

# Validation data generator
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    'dataset/validation',  # Adjust this path to your dataset
    target_size=(256, 256),
    batch_size=16,
    class_mode='categorical',
    shuffle=False)  # Important: keep data in same order as labels

# Correct the step size for model prediction
steps = int(np.ceil(validation_generator.samples / validation_generator.batch_size))

# Generate predictions for the validation set
predictions = model.predict(validation_generator, steps=steps)

# Get the index of the highest confidence predictions
predicted_classes = np.argmax(predictions, axis=1)

# Ensure the true_classes and predicted_classes have the same length
true_classes = validation_generator.classes[:len(predicted_classes)]
class_labels = list(validation_generator.class_indices.keys())

# Handle case where number of predictions does not match number of labels
unique_predicted_classes = np.unique(predicted_classes)
labels = [class_labels[i] for i in unique_predicted_classes]

# Calculate and print the metrics
accuracy = accuracy_score(true_classes, predicted_classes)
f1 = f1_score(true_classes, predicted_classes, average='weighted', labels=unique_predicted_classes)
precision = precision_score(true_classes, predicted_classes, average='weighted', labels=unique_predicted_classes)
recall = recall_score(true_classes, predicted_classes, average='weighted', labels=unique_predicted_classes)

print(f'Accuracy: {accuracy:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')

# Print detailed classification report
print('\nClassification Report\n')
print(classification_report(true_classes, predicted_classes, target_names=labels))

