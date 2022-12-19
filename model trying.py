import tensorflow as tf
from keras.layers import Conv2D,Flatten,BatchNormalization,MaxPool2D, GlobalAveragePooling2D
from keras.applications.resnet import preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.applications.resnet import ResNet50
from keras.preprocessing import image
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np

#https://www.youtube.com/watch?v=1Gbcp66yYX4
#here we are providing the height and width of the input layer for the ResNet50
img_height, img__width = (224,244)
batch_size = 32

train_data_dir = r"processed_stuff/train"
test_data_dir = r"processed_stuff/test"
val_data_dir = r"processed_stuff/val"

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=0.4)
train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_height, img__width),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    subset='training')
valid_generator = train_datagen.flow_from_directory(val_data_dir,
                                                      target_size=(img_height,img__width),
                                                      batch_size=batch_size,
                                                      class_mode='categorical',
                                                      subset='validation')
test_generator=train_datagen.flow_from_directory(test_data_dir,
                                                 target_size=(img_height,img__width),
                                                 batch_size=1,
                                                 class_mode='categorical',
                                                 subset='validation')
x,y = test_generator.next()
base_model = ResNet50(include_top=False, weights='imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense