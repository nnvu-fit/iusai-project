import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import KFold
import os

class EarlyStoppingAtAccuracy(Callback):
    def __init__(self, accuracy=0.99):
        super(EarlyStoppingAtAccuracy, self).__init__()
        self.accuracy = accuracy

    def on_epoch_end(self, epoch, logs=None):
        if logs.get("accuracy") >= self.accuracy:
            print(f"/nReached {self.accuracy * 100}% accuracy so stopping training!")
            self.model.stop_training = True

train_dir = 'subjects-small/'
generator = ImageDataGenerator()
train_ds = generator.flow_from_directory(train_dir, target_size=(224, 224), batch_size=50, shuffle=False)
print(train_ds)

# Load all images and labels
all_images = []
all_labels = []

for i in range(len(train_ds)):
    #print(batch)
    batch = train_ds[i]
    images, labels = batch
    all_images.append(images)
    all_labels.append(labels)

all_images = np.vstack(all_images)
all_labels = np.vstack(all_labels)

def get_classes(image_dir):
    return [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]

num_classes = len(get_classes(train_dir))

def build_model(num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(96, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model

kfold = KFold(n_splits=5, shuffle=True)
cvscores = []
best_accuracy = 0
best_model = None
for train, test in kfold.split(all_images, all_labels):
    model = build_model(num_classes)
    early_stopping_callback = EarlyStoppingAtAccuracy(accuracy=0.99)
    history = model.fit(all_images[train], all_labels[train],
                        epochs=30,
                        batch_size=60,
                        validation_data=(all_images[test], all_labels[test]),
                        callbacks=[early_stopping_callback])
    scores = model.evaluate(all_images[test], all_labels[test], verbose=0)
    print(f"{model.metrics_names[1]}: {scores[1] * 100:.2f}%")
    cvscores.append(scores[1] * 100)
    
    # Save model if it has the best accuracy so far
    if scores[1]> best_accuracy:
        best_accuracy = scores[1]
        best_model = model
        model.save('best_model.h5')
        print(f"Best model saved with accuracy: {best_accuracy * 100:.2f}%")

print(f"Mean accuracy: {np.mean(cvscores):.2f}% (+/- {np.std(cvscores):.2f}%)")

