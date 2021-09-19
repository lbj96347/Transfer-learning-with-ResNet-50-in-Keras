# Transfer-learning-with-ResNet-50-in-Keras 

Introdution
------------

This project is aiming to train a image classification model by transfer learning with ResNet50 pre-trained model. It is impelemented by Keras. 

Requirements
------------

*conda* for open source package management and environment management 

h5py==2.10.0

matplotlib

keras==2.2.5

tensorflow==1.14

Usage
------

### Step1: Install conda  

visit [conda installation](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) & install the compatible version on your system environment. 

### Step 2: Create virtual environment for this project 

> conda create -n tl-resnet50 keras==2.2.5

### Step 3: Activate this env

> conda activate tl-resnet50 

### Step 4: Install requirements 

> pip install -r requirements 

### Step 5: Create folders for storing data & models 

> mkdir data && mkdir models 

### Step 6: Put your dataset into data folder 

Your folder structure should be like the following format 

```
data > 
  your_dataset > 
    train > 
      class_1 > 
        img_1.jpg
        img_2.jpg
        img_3.jpg
        ...
      class_2 >
        img_1.jpg
        img_2.jpg
        img_3.jpg
        ...
    validation >
      class_1 > 
        img_1.jpg
        img_2.jpg
        img_3.jpg
        ...
      class_2 >
        img_1.jpg
        img_2.jpg
        img_3.jpg
        ...
```

### Step 7: Replace 'input_path' value in train.py & predict.py

in both file 

```
input_path = './data/your_dataset/'
```

in predict.py 

```
validation_img_paths = ['./validation/img_1.jpg', './validation/img_2.jpg', './validation/img_3.jpg']
```

## Step 8: Train model & Validate training model 

Training model:  

> python train.py 

Validate training result:

> python predict.py 


