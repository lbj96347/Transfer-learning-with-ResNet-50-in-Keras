import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras import Model, layers
from keras.models import load_model, model_from_json

input_path = "[YOUR_TRAINING_DATASET_PATH]"

def predict_imgs():
    model = load_model('./models/keras/model.h5')

    validation_img_paths = ["YOUR_VALIDATION_DATASET_PATHS"]

    img_list = [Image.open(input_path + img_path) for img_path in validation_img_paths]

    validation_batch = np.stack([preprocess_input(np.array(img.resize((224,224))))
                             for img in img_list])

    pred_probs = model.predict(validation_batch)

    fig, axs = plt.subplots(1, len(img_list), figsize=(20, 5))

    for i, img in enumerate(img_list):
        ax = axs[i]
        ax.axis('off')
        title = "{:.0f}% class_1, {:.0f}% class_2".format(100*pred_probs[i,0],
                                                                100*pred_probs[i,1]) 
        print(title)
        ax.set_title(title)
        ax.imshow(img)
    plt.show()

if __name__ == '__main__':
    predict_imgs()
