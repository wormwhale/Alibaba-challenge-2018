
# coding: utf-8

# In[92]:


# import package
import numpy as np
import functools
import pandas as pd
import os
import copy
import operator

# this is a function I gave
import path_finding as pf


# In[93]:


#read city data
df_city = pd.read_csv('./data/CityData.csv')
#get file name
file_list=os.listdir('./data/partition')
del file_list[0]
#result list################################  delete once got jingyu's result
result_list=os.listdir('./data/predict_wind')
del result_list[0]
############################################
#print file_list
#df = pd.read_csv('./data/partition/'+file_list[0])
#predict=pd.read_csv('./data/predict_wind/'+file_list[0])
#range(0,len(file_list))


# In[95]:



error=[]
path_result=pd.DataFrame([])
path_result.to_csv('path_result.csv', header=False, index=False)
## i : file number
for i in range(0,len(file_list)):
    # loading one-day data
    df = pd.read_csv('./data/partition/'+file_list[i])
    ###########################################  unmark once got jingyu's result
    #predict_wind=predict()
    ###########################################  delete once got jingyu's result
    predict_wind=pd.read_csv('./data/predict_wind/'+result_list[i])
    ############################################
    
    # change time scale 
    predict_wind['hour'] = predict_wind['hour']*60-180
    ## j : cid (city id) 
    for j in df_city.cid[1:11]:
    #for j in df_city.cid[1:len(df_city)]:
        #because every file have same date_id, I only get the first one to use
        dateid=predict_wind.date_id[0]
        try:
            path_result=pf.create_path(predict_wind,df_city,j,dateid)
        except Exception as er:
            error.append([dateid,j,er])        
    #write result to csv        
        with open('./output/path_result.csv', 'a') as f:
            (path_result).to_csv(f, header=False, index=False)

error=pd.DataFrame(error)
with open('./output/error_log.csv', 'a') as f:
    (error).to_csv(f, header=False, index=False)
# load the city information

