#this is pandas test tutorial

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/school_earnings.csv')
print(data)
print(data.describe())
data.info()
data.isnull().sum()
data['School'].value_counts()
#plotting
plot=data['School'].value_counts().plot(kind='bar')
plt.show()
data['School'].value_counts().plot(kind='barh')
data['School'].value_counts().plot(kind='pie')
plt.show()
data['School'].value_counts().plot(kind='box')
plt.show()
data['School'].value_counts().plot(kind='hist')
plt.show()
data['School'].value_counts().plot(kind='area')
plt.show()
data.plot(kind='scatter', x='Women', y='Men')
plt.show()
data['School'].value_counts().plot(kind='pie')
data['School'].value_counts().plot(kind='line')
plt.show()
data['School'].value_counts().plot(kind='bar')
