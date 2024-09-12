#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[20]:


dtype_dict = {23: str, 24: str, 70: str, 71: str}
accident_data = pd.read_csv('veh_16.csv', dtype=dtype_dict,low_memory=False)


# In[21]:


accident_data


# In[27]:


road_conditions = accident_data.groupby('CASENUM').size().reset_index(name='Count')


print('Accidents by Road Condition:')
print(road_conditions)


plt.figure(figsize=(10, 6))
plt.bar(road_conditions['CASENUM'], road_conditions['Count'])
plt.xlabel('Road Condition')
plt.ylabel('Number of Accidents')
plt.title('Accidents by Road Condition')
plt.xticks(rotation=90)
plt.show()


# In[28]:


weather_conditions = accident_data.groupby('REGION').size().reset_index(name='Count')

print('\nAccidents by Weather Condition:')
print(weather_conditions)

plt.figure(figsize=(10, 6))
plt.plot(weather_conditions['REGION'], weather_conditions['Count'])
plt.xlabel('Region')
plt.ylabel('Number of Accidents')
plt.title('Accidents by Weather Condition')
plt.xticks(rotation=90)
plt.show()


# In[29]:


accident_data['Time_of_Day'] = pd.cut(accident_data['PSUSTRAT'], bins=[0, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'])
time_of_day = accident_data.groupby('Time_of_Day', observed=False).size().reset_index(name='Count')
print('\nAccidents by Time of Day:')
print(time_of_day)

plt.figure(figsize=(8, 6))
plt.pie(time_of_day['Count'], labels=time_of_day['Time_of_Day'], autopct='%1.1f%%')
plt.title('Accidents by Time of Day')
plt.axis('equal')  
plt.show()


# In[30]:


accident_locations = accident_data.groupby(['URBANICITY', 'V_ALCH_IM']).size().reset_index(name='Count')

plt.figure(figsize=(10, 6)) 
plt.scatter(accident_locations['V_ALCH_IM'], accident_locations['URBANICITY'], s=accident_locations['Count'], alpha=0.5)
plt.xlabel('V_ALCH_IM')
plt.ylabel('URBANICITY')
plt.title('Accident Hotspots')
cbar = plt.colorbar()
cbar.set_label('Number of Accidents')
plt.show()


# In[ ]:




