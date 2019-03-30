
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn-white') #for jupyter dark, change style
%matplotlib inline
%config InlineBackend.figure_format = 'svg' #Graphics in SVG format are more sharp and legible
import scipy.sparse
import warnings
import random

#Load data in dataframe
pd.set_option('display.max_columns', None) 
iowa_liquor = pd.read_csv(r"data/vodka/vodka.csv")
iowa_liquor.head()

iowa_liquor.shape

iowa_liquor.dtypes

iowa_liquor.describe()

# count of duplicates
iowa_liquor.duplicated().sum()

# display the duplicates now
iowa_liquor[iowa_liquor.duplicated()]

#iowa_liquor['Store Number'].nunique()
iowa_liquor.duplicated('Store Number')

# printing out column names
iowa_liquor.columns

# find summary statistics for each numerical column
iowa_liquor.describe()

#Creates a county labels df
county=['County Number','County']
cn=iowa_liquor[county].dropna()
cn = cn.groupby(by=['County Number'], as_index=False)
cn = cn.agg({'County': lambda x: x.iloc[0]})
cn.to_csv('cn.csv')

#see if there are null values in the dataset
iowa_liquor.isnull().sum(axis = 0)

# drop spaces in the name columns
df = iowa_liquor.copy()
iowa_liquor.columns = [c.replace('/','_').replace(' ','_').replace(')','').replace('(','').lower() for c in df.columns.tolist()]

#Gets rid of rows with empty data
df_new = iowa_liquor.dropna(axis=0, how='any')

#drops to $ from the pricing data
cols_with_dollar = ['state_bottle_cost','state_bottle_retail','sale_dollars']
for col in cols_with_dollar:
    df_new[col] = df_new[col].apply(lambda x: x.strip('$')).astype('float')
df_new.head()

#Converts to date times
df_new["date"] = pd.to_datetime(df_new["date"], infer_datetime_format=True)

#Turns floats into integers
intconv = lambda x: int(x)
df_new['county_number']= df_new['county_number'].apply(intconv)
df_new['category']= df_new['category'].apply(intconv)

#Turns objects into floats
floatconv = lambda x: float(x)
df_new['sale_dollars']= df_new['sale_dollars'].apply(floatconv)
df_new['state_bottle_cost']= df_new['state_bottle_cost'].apply(floatconv)
df_new['state_bottle_retail']= df_new['state_bottle_retail'].apply(floatconv)

#Turns int into floats
floatconv = lambda x: float(x)
df_new['store_number']= df_new['store_number'].apply(floatconv)

df_new.dtypes

df_new = pd.DataFrame(df_new, columns =['invoice_item_number', 'date',
              'store_number', 'store_name', 'address', 'city',
              'zip_code', 'county_number',
              'county', 'category', 'category_name', 'vendor_number',
              'vendor_name', 'item_number', 'item_description',
              'pack', 'bottle_volume_ml','state_bottle_cost', 
              'state_bottle_retail', 'bottles_sold','sale_dollars',
              'volume_sold_liters', 'volume_sold_gallons'
                        ])
df_new.head()

#save file as csv
df_new.to_csv("data/vodka/vodka_modeler.csv",index=False, sep='|')

