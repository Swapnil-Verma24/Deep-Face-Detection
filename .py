import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


IMAGE_SIZE = (120, 120)
BATCH_SIZE = 32
EPOCHS = 50
DATA_DIR = 'data'  


def load_data(data_dir):
    
    datagen = ImageDataGenerator(validation_split=0.2, 
                                  rescale=1./255, 
                                  rotation_range=20,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',  
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',  
        subset='validation'
    )

    return train_generator, validation_generator


def build_model():
    base_model = tf.keras.applications.VGG16(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
                                               include_top=False,
                                               weights='imagenet')
    
    
    base_model.trainable = False

    
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def main():
    train_generator, validation_generator = load_data(DATA_DIR)
    model = build_model()
    
    
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS
    )
    
    
    model.save('face_detection_model.h5')
    print("Model trained and saved as 'face_detection_model.h5'")

if __name__ == "__main__":
    main()
