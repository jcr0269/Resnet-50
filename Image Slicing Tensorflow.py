import os
import tensorflow as tf
import nib

get_list=[]
image_data_path = 'flowers/daisy'
for i in sorted(os.listdir(image_data_path)):
    # print(i)
    get_list.append(i)
    dataset = tf.data.Dataset.from_tensor_slices(get_list)
