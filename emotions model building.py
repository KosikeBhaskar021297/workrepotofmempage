#!/usr/bin/env python
# coding: utf-8

# In[16]:


import matplotlib.pyplot as plt


# In[1]:


#get_ipython().system('pip install numpy pandas tensorflow')


# In[2]:


import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[3]:


data_path=r"C:\Users\DELL\Downloads\facial emotions dataset\images\images"


# In[1]:


EMOTIONS = ["angry","disgust","fear","graypout","happy","neutral","sad","surprise","wynk"]


# In[2]:


EMOTIONS[5]


# In[4]:


img_height,img_width= 64,64


# In[5]:


datagen=ImageDataGenerator(rescale=1./255,rotation_range=20,width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True)


# In[6]:


train_generator = datagen.flow_from_directory(
    os.path.join(data_path, "train"),
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode="categorical"
)


# In[7]:


validation_generator = datagen.flow_from_directory(
    os.path.join(data_path, "validation"),
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode="categorical"
)


# In[8]:


model = Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(img_height, img_width, 1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(9, activation="softmax")
])


# In[9]:


model.summary()


# In[10]:


model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# In[11]:


model.fit(train_generator,epochs=15,validation_data=validation_generator)


# In[14]:


model.save("swapna.h5")


# In[21]:


from tensorflow.keras.preprocessing import image
img_path='C:/Users/DELL/Downloads/facial emotions dataset/images/images/validation/happy/80.jpg'
test_image=image.load_img(img_path,target_size=(48,48,1),color_mode='grayscale')
test_image=image.img_to_array(test_image)
print(test_image.shape)
plt.imshow(test_image)


# In[ ]:




