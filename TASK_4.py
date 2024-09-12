#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


import os


# In[3]:


import cv2


# In[1]:


gesture_c = r"D:\Prodigy internship\archive (1)\leapGestRecog\leapGestRecog\01\gesture_c"
gesture_down = r"D:\Prodigy internship\archive (1)\leapGestRecog\leapGestRecog\01\gesture_down"
gesture_fist = r"D:\Prodigy internship\archive (1)\leapGestRecog\leapGestRecog\01\gesture_fist"
gesture_fist_moved = r"D:\Prodigy internship\archive (1)\leapGestRecog\leapGestRecog\01\gesture_fist_moved"
gesture_index = r"D:\Prodigy internship\archive (1)\leapGestRecog\leapGestRecog\01\gesture_index"
gesture_I = r"D:\Prodigy internship\archive (1)\leapGestRecog\leapGestRecog\01\gesture_l"
gesture_ok = r"D:\Prodigy internship\archive (1)\leapGestRecog\leapGestRecog\01\gesture_ok"
gesture_palm = r"D:\Prodigy internship\archive (1)\leapGestRecog\leapGestRecog\01\gesture_palm"
gesture_palm_moved = r"D:\Prodigy internship\archive (1)\leapGestRecog\leapGestRecog\01\gesture_palm_moved"
gesture_thumb = r"D:\Prodigy internship\archive (1)\leapGestRecog\leapGestRecog\01\gesture_thumb"


# In[2]:


data = []
label = []
size = (64,64)


# In[6]:


for img in os.listdir(gesture_c):
    img1_path = os.path.join(gesture_c,img)
    img1 = cv2.imread(img1_path)
    if img1 is not None:
        img1_size = cv2.resize(img1,size)
        img1_color = cv2.cvtColor(img1_size,cv2.COLOR_BGR2GRAY)
        data.append(img1_color.flatten())
        label.append("c type finger")

for img in os.listdir(gesture_down):
    img2_path = os.path.join(gesture_down,img)
    img2 = cv2.imread(img2_path)
    if img2 is not None:
        img2_size = cv2.resize(img2,size)
        img2_color = cv2.cvtColor(img2_size,cv2.COLOR_BGR2GRAY)
        data.append(img2_color.flatten())
        label.append("down type finger")

for img in os.listdir(gesture_fist):
    img3_path = os.path.join(gesture_fist,img)
    img3 = cv2.imread(img3_path)
    if img3 is not None:
        img3_size = cv2.resize(img3,size)
        img3_color = cv2.cvtColor(img3_size,cv2.COLOR_BGR2GRAY)
        data.append(img3_color.flatten())
        label.append("fist type finger")

for img in os.listdir(gesture_fist_moved):
    img4_path = os.path.join(gesture_fist_moved,img)
    img4 = cv2.imread(img4_path)
    if img1 is not None:
        img4_size = cv2.resize(img4,size)
        img4_color = cv2.cvtColor(img4_size,cv2.COLOR_BGR2GRAY)
        data.append(img4_color.flatten())
        label.append("fist_moved type finger")

for img in os.listdir(gesture_index):
    img5_path = os.path.join(gesture_index,img)
    img5 = cv2.imread(img5_path)
    if img5 is not None:
        img5_size = cv2.resize(img5,size)
        img5_color = cv2.cvtColor(img5_size,cv2.COLOR_BGR2GRAY)
        data.append(img5_color.flatten())
        label.append("index type finger")

for img in os.listdir(gesture_I):
    img6_path = os.path.join(gesture_I,img)
    img6 = cv2.imread(img6_path)
    if img6 is not None:
        img6_size = cv2.resize(img6,size)
        img6_color = cv2.cvtColor(img6_size,cv2.COLOR_BGR2GRAY)
        data.append(img6_color.flatten())
        label.append("I type finger")

for img in os.listdir(gesture_ok):
    img7_path = os.path.join(gesture_ok,img)
    img7 = cv2.imread(img7_path)
    if img7 is not None:
        img7_size = cv2.resize(img7,size)
        img7_color = cv2.cvtColor(img7_size,cv2.COLOR_BGR2GRAY)
        data.append(img7_color.flatten())
        label.append("ok type finger")

for img in os.listdir(gesture_palm):
    img8_path = os.path.join(gesture_palm,img)
    img8 = cv2.imread(img8_path)
    if img8 is not None:
        img8_size = cv2.resize(img8,size)
        img8_color = cv2.cvtColor(img8_size,cv2.COLOR_BGR2GRAY)
        data.append(img8_color.flatten())
        label.append("palm type finger")

for img in os.listdir(gesture_palm_moved):
    img9_path = os.path.join(gesture_palm_moved,img)
    img9 = cv2.imread(img9_path)
    if img9 is not None:
        img9_size = cv2.resize(img9,size)
        img9_color = cv2.cvtColor(img9_size,cv2.COLOR_BGR2GRAY)
        data.append(img9_color.flatten())
        label.append("palm_moved type finger")

for img in os.listdir(gesture_thumb):
    img10_path = os.path.join(gesture_thumb,img)
    img10 = cv2.imread(img10_path)
    if img10 is not None:
        img10_size = cv2.resize(img10,size)
        img10_color = cv2.cvtColor(img10_size,cv2.COLOR_BGR2GRAY)
        data.append(img10_color.flatten())
        label.append("thumb type finger")


# In[7]:


data


# In[9]:


label


# In[10]:


data = np.array(data)
label = np.array(label)


# In[11]:


data


# In[12]:


label


# In[15]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[29]:


x_train,x_test,y_train,y_test = train_test_split(data,label,test_size = 0.3, random_state = 42,stratify=label)


# In[32]:


prediction = rf.predict(x_test)
prediction


# In[33]:


accuracy = accuracy_score(prediction , y_test)
accuracy


# In[34]:


print("The accuracy_score of hand gesture model is 1.0 , So it is the best predicted model for hand gesture machine")


# In[ ]:





# In[ ]:




