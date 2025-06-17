#imports
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import cv2
from sklearn.metrics import classification_report
#loading data
x = []
labels = []
y = []
filepath_testing = '/content/drive/MyDrive/Datasets/Single label classification/testing'
filepath_training = '/content/drive/MyDrive/Datasets/Single label classification/training'
labels = {'forks':[1,0,0,0],'glasses':[0,1,0,0], 'plates':[0,0,1,0],'spoons':[0,0,0,1]}
for item in tqdm(labels.keys()):
  for i in range(40):
    img_path = os.path.join(filepath_training,item)
    ch = str(i+1)+'.png'
    if i+1<10:
      ch = '0'+ch
    img_path = os.path.join(img_path,ch)
    image = Image.open(img_path)
    image = np.array(image)
    image = image/255
    image = cv2.resize(image,(256,256))
    x.append(image)
    y.append(labels[item])
for item in tqdm(labels.keys()):
  for i in range(10):
    img_path = os.path.join(filepath_testing,item)
    ch = str(i+1)+'.png'
    if i+1<10:
      ch = '0'+ch
    img_path = os.path.join(img_path,ch)
    image = Image.open(img_path)
    image = np.array(image)
    image = image/255
    image = cv2.resize(image,(256,256))
    x.append(image)
    y.append(labels[item])
x = np.array(x)
y = np.array(y)
#splitting data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=90)
#building model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),input_shape = (256,256,3),activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dense(4,activation='softmax')
])
#compiling model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#training
model.fit(x_train,y_train,epochs=7,batch_size=5)
#evaluation
model.evaluate(x_test,y_test)
#full report
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred,axis=1)
y_true = np.argmax(y_test,axis=1)
print(classification_report(y_pred,y_true,digits=2))
#saving and loading model
model.save("/content/drive/MyDrive/Datasets/Single label classification/my_model.keras") 
#model = tf.keras.models.load_model("/content/drive/MyDrive/Datasets/Single label classification/my_model.keras")
