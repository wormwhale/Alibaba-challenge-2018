
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
train1=pd.read_csv('ForecastDataforTraining_201712_1.csv')
city=pd.read_csv('In_situMeasurementforTraining_201712.csv')


# In[ ]:




# In[ ]:




# In[ ]:




# In[3]:

day1 = pd.merge(train1, city,  how='left', on=['xid','yid','date_id','hour'])

day1.head()


# In[ ]:




# In[ ]:




# In[4]:

day1_x_train=np.empty((4152744,10))
day1_y_train=np.empty(4152744)
for i in range(4152744):
    day1_x_train[i]=day1['wind_x'].values[i*10:(i+1)*10]
    day1_y_train[i]=day1['wind_y'].values[i*10]

print day1_x_train,day1_y_train


# In[20]:

(day1_y_train>=15.0).astype(int)


# In[21]:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(day1_x_train, (day1_y_train>=15.0).astype(int), test_size=0.25, random_state=42)


# In[7]:

import os
os.environ['KERAS_BACKEND']='theano'
from keras.models import Sequential


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[9]:


from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop


# In[ ]:




# In[22]:

model = Sequential([
    Dense(40, input_dim=10),  
    Activation('softmax'),
    Dense(units=20,activation='softmax'),
    Dense(1),
    Activation('sigmoid'),
])

#rmsprop = RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer="rmsprop",
              loss='binary_crossentropy',
              metrics=['accuracy'])
print('Training ------------')
model.fit(X_train, y_train, epochs=10, batch_size=1000)
print('\nTesting ------------')

loss, accuracy = model.evaluate(X_test, y_test)

print('test loss: ', loss)
print('test accuracy: ', accuracy)


# In[ ]:



