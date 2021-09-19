import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
from PIL import Image

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras import Model, layers
from keras.models import load_model, model_from_json

print(keras.__version__)# should be 2.2.5

import tensorflow as tf
print(tf.__version__)  # should be 1.14.x

import PIL
print(PIL.__version__) # should be 5.2.0

input_path = "[YOUR_TRAINING_DATASET_PATH]"

def init_train_generator():
    train_datagen = ImageDataGenerator(
        shear_range=10,
        zoom_range=0.2,
        horizontal_flip=True,
        preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        input_path + 'train',
        batch_size=32,
        class_mode='binary',
        target_size=(224,224))
    return train_generator

def init_validation_generator():
    validation_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input)
    validation_generator = validation_datagen.flow_from_directory(
        input_path + 'validation',
        shuffle=False,
        class_mode='binary',
        target_size=(224,224))
    return validation_generator  

# Use ResNet50 
def use_resnet50():
    conv_base = ResNet50(
        include_top=False,
        weights='imagenet')
    for layer in conv_base.layers:
        layer.trainable = False
    x = conv_base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x) 
    predictions = layers.Dense(2, activation='softmax')(x)
    model = Model(conv_base.input, predictions)
    optimizer = keras.optimizers.Adam()
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    resnet50model = model
    return resnet50model


def start_to_train():
    train_generator = init_train_generator()
    validation_generator = init_validation_generator()
    model = use_resnet50()
    history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=347 // 32,  # added in Kaggle
                              epochs=3,
                              validation_data=validation_generator,
                              validation_steps=10  # added in Kaggle
                             )
    model.save('models/keras/model.h5')
    '''
    model.save_weights('models/keras/weights.h5')
    with open('models/keras/architecture.json', 'w') as f:
        f.write(model.to_json())
    '''

if __name__ == '__main__' :
    # args = parse_args()
    start_to_train()

