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


df_new= pd.read_csv("data/vodka/vodka_modeler.csv", sep='|')

# the 'count' row is not very useful and the store_number column is meaningless in this table
df_new.describe().iloc[1:,1:]

# Histograms
def draw_histograms(df,col,bins):
    df[col].hist(bins=bins);
    plt.title(col);
    plt.xlabel(col);
    plt.xticks(rotation=90);
    plt.show();
    print
cols = ['bottles_sold', 'sale_dollars', 'volume_sold_liters']
for col in cols:
    draw_histograms(df_new, col,bins=10)
    

## There are some obvious problems: In the columns bottle_volume_ml and volume_sold_liters there are zero values which can be dropped.
## The maximum values are many $\sigma$s larger than the mean for all columns, indicating outliers. Let us drop some outliers. 
## We will choose to restrict some of the columns. We will drop values roughly 2$\sigma$ larger than the means.   

#Exclude outliers
df1 = df_new.copy()
print(np.mean(df1['bottles_sold'])+ 2*np.std(df1['bottles_sold']))
print(np.mean(df1['sale_dollars'])+ 2*np.std(df1['sale_dollars']))
print(np.mean(df1['volume_sold_liters'])+ 2*np.std(df1['volume_sold_liters']))

df1 = df_new.copy()
cutoffs = (df1['bottle_volume_ml'] > 0) & (df1['bottles_sold'] < 63.443762273733725) & (df1['volume_sold_liters'] < 79.51446145172659) & (df1['sale_dollars'] < 817.8652993794766)
df1 = df1[cutoffs]

df1.describe()

df1.describe().iloc[1:,1:]

cols = ['bottles_sold', 'sale_dollars', 'volume_sold_liters']
for col in cols:
    draw_histograms(df1, col,bins=10)
    
df1.to_csv('data/vodka/lowa_liquor_sales.csv')

# determine unique values in a column
df1.nunique()      # count the number of unique values in each column

# Creating new column of net profit per bottle for each bottle
df1['state_profit_per_bottle'] = df1['state_bottle_retail'] - df1['state_bottle_cost']

# Grouping by vendor 
df1.groupby(['vendor_number'])[['state_bottle_cost', 'state_bottle_retail']].describe()

#bar graph showing top 10 counties with most sales
plt.figure(figsize=(8,4))
plt.style.use('fivethirtyeight')
ax=sns.barplot(x = 'sale_dollars', y = 'county', data = df1)
ax.set(ylabel='County', xlabel='Sales Amount ($)' , title='Liquor Sales by County')

# most popular categories featured 
pop_categories = df1['category_name'].value_counts()
pop_categories.head(25).plot(kind='barh', width=0.4, figsize=(2,15))

print(df1['state_profit_per_bottle'].describe())

# histogram sales data 
# only going to 350 bc thats where most of data is clustered, showing more of the graph will not showcase data clearly
plt.hist(df1['sale_dollars'], bins=1000)
plt.xlabel('Sale ($)')
plt.ylabel('Frequency')
plt.title('Histogram of Sales')
plt.xlim([0, 350])

#Distribution of target variable: sale_dollars
plt.style.use('fivethirtyeight')
hist = plt.figure(figsize=(12,7))
sns.distplot(df1.sale_dollars, bins = 25)
plt.xlabel("Sales in dollars"), plt.ylabel("Number of Buyers")
plt.title("Sales amount Distribution")
hist.savefig(r'data/vodka/Hist_Sales.png')

#bar graph showing top 10 liquors sold
liquor = plt.figure(figsize=(8,5))
plt.style.use('fivethirtyeight')
ax=sns.barplot(x = 'sale_dollars', y = 'category_name', data = df1)
ax.set(ylabel='Liquor Type', xlabel='Sales Amount ($)' , title='Liquor Sales by Type')
liquor.savefig(r'data/vodka/LiquorType_Sales.png')

# Final dataframe by zip codes
zcode_table = df1[['store_number','zip_code', 'city', 'bottle_volume_ml',
                  'state_bottle_cost', 'state_bottle_retail',
                  'bottles_sold','sale_dollars','volume_sold_liters',
                  'state_profit_per_bottle', 'county'
                        ]]
                  
#correlation between numerical values and target variable: sale_dollars
corr = zcode_table.corr()
print (corr['sale_dollars'].sort_values(ascending=False)[:15])

#correlation matrix
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corr, vmax=.8, annot_kws={'size': 20}, annot=True);

# plotting relationship to see if any outliers exist
plt.scatter(x=zcode_table['bottles_sold'], y=zcode_table['volume_sold_liters'], marker='+')
plt.xlabel('Bottles Sold')
plt.ylabel('Volume Sold (Liters)')

# plotting relationship to see if any outliers exist
plt.scatter(x=zcode_table['bottles_sold'], y=zcode_table['state_profit_per_bottle'], marker='+') # interesting relationship
plt.xlabel('Bottles Sold') 
plt.ylabel('State Profit Per Bottle')

zcode_table.head()
zcode_table.shape[0]

# Forming a new dataframe, grouped by zip code, and then the sum of Sale (Dollars) for each zip code.
# This will tell me what zip codes are paying the most for liquor.

# stores by zip
stores_by_zip = zcode_table.groupby('zip_code')['store_number'].nunique().to_frame().sort_values('store_number', ascending=False)
stores_by_zip['zip_code'] = stores_by_zip.index
stores_by_zip.index = range(0, len(stores_by_zip))
stores_by_zip.head()

#sales by zip
sales_by_zip = zcode_table.groupby('zip_code')['sale_dollars'].sum().to_frame().sort_values('sale_dollars', ascending=0)
sales_by_zip['zip_code'] = sales_by_zip.index

# creating an index of range 0 - length of the 
sales_by_zip.index = range(0, len(sales_by_zip))
print(sales_by_zip.shape)
sales_by_zip.head()

#volume sold liters by zip 
# Forming a dataframe of the volume distributed to per zip code
volume_by_zip = zcode_table.groupby('zip_code')['volume_sold_liters'].sum().to_frame()
volume_by_zip['zip_code'] = volume_by_zip.index
volume_by_zip.index = range(0, len(volume_by_zip))
volume_by_zip.head()

#Bottle profit per zip code  (net profit per bottle for each bottle  'state_bottle_retail' -'state_bottle_cost')
bottleprofit_per_zip = zcode_table.groupby('zip_code')['state_profit_per_bottle'].sum().to_frame()
bottleprofit_per_zip['zip_code'] = bottleprofit_per_zip.index
bottleprofit_per_zip.index = range(0, len(bottleprofit_per_zip))
bottleprofit_per_zip.head()

# Bottles sold by zip 
bottles_by_zip = zcode_table.groupby('zip_code')['bottles_sold'].sum().to_frame()
bottles_by_zip['zip_code'] = bottles_by_zip.index
bottles_by_zip.index = range(0, len(stores_by_zip))
bottles_by_zip.head()

#### Rename Number stores by zip 
#renaming column to appropriate description
stores_by_zip.rename(columns={'store_number' : 'Number of Stores Per zip'}, inplace=True)
stores_by_zip.head()

#### Merging data sets in zip_frame
# merging stores and sales dataframe to matching zip codes
zip_frame = pd.merge(stores_by_zip, sales_by_zip, how='inner', on='zip_code')
print(zip_frame.shape)
zip_frame.head()

# merging volume frame 
zip_frame = pd.merge(zip_frame, volume_by_zip, how='inner', on='zip_code')
print(zip_frame.shape)
zip_frame.head()

# mergine profit per bottle frame
zip_frame = pd.merge(zip_frame, bottleprofit_per_zip, how='inner', on='zip_code')
print(zip_frame.shape)
zip_frame.head()

# merging bottles per zip as well 
zip_frame = pd.merge(zip_frame, bottles_by_zip, how='inner', on='zip_code')
print(zip_frame.shape)
zip_frame.head()

# Feature Engineering
zip_frame['sales_per_store'] = zip_frame['sale_dollars'] / zip_frame['Number of Stores Per zip']
zip_frame.head()

zip_frame.rename(columns={'sale_dollars' : 'total_sales'}, inplace=True)
zip_frame.head()

zip_frame.shape[0]

zip_frame.dtypes

zip_frame['zip_code'] = zip_frame.zip_code.astype(str)

#### Adding demographic data to zip_frame 
# Population data
pop = pd.read_csv(r"data/vodka/pop_iowa_per_zip.csv")
# to change use .astype() 
pop['zip_code'] = pop.zip_code.astype(str)
#pop['total_population'] = pop.total_population.astype(str)

pop.head()
print(pop.shape[0])
#Merge pop and vodka
zip_frame= pd.merge(zip_frame, pop, on= 'zip_code', how='left')

print(zip_frame.shape[0])

#Calculation of ratios: stores per person and alcohol consumption per person
zip_frame['store_population_ratio'] = zip_frame['Number of Stores Per zip']/zip_frame['total_population']
zip_frame['consumption_per_capita'] = zip_frame['volume_sold_liters']/zip_frame['total_population']

#area data
pd.set_option('display.max_rows', None)
areas = pd.read_csv(r"data/vodka/ia_zip_city_county_sqkm.csv")
areas.columns = ['zip_code', 'city', 'county', 'state', 'county_number', 'area']
areas = areas[(areas['county'] != 'Polk') & (areas['county'] != 'Fremont')]
areas = areas[['zip_code','area']].groupby(['zip_code'])[['area']].sum()
areas.reset_index(level=[0], inplace=True)
areas['zip_code'] = areas.zip_code.astype(str)
areas

areas.dtypes

#Merge area and vodka
#zcode_table = pd.merge(zcode_table, areas, on= 'county', how='outer')
zip_frame = pd.merge(zip_frame, areas, on= 'zip_code', how='left')

zip_frame['stores_per_area'] = zip_frame['Number of Stores Per zip']/ zip_frame['area']

#Income data
pd.set_option('display.max_rows', None)
income = pd.read_excel(r"data/vodka/zctaincome2000.xls")
income.columns = ['zip_code', 'Median_Household_Income', 'Median_Family_Income', 'Per_Capita_Income']
#income.reset_index(level=[0], inplace=True)
income['zip_code'] = income.zip_code.astype(str)
income

#Merge Income and vodka
zip_frame= pd.merge(zip_frame, income, on= 'zip_code', how='left')

# Compute log of total sales
zip_frame['log_total_sales'] =np.log(zip_frame['total_sales'])

zip_frame['log_total_sales_centered'] = zip_frame['log_total_sales'] - np.mean(zip_frame['log_total_sales'])

# FINAL DATA FOR MODELING AND CLASSIFICATION
# Re-forming the dataframe to only inlcude the top 100 selling zip codes.
zip_frame = zip_frame.sort_values('total_sales', ascending=0)

# Selecting variables
zip_frame = zip_frame[['Number of Stores Per zip', 'zip_code', 'total_sales', 'volume_sold_liters', 'state_profit_per_bottle',
                      'sales_per_store', 'bottles_sold', 'store_population_ratio', 'consumption_per_capita', 'stores_per_area',
                       'median_age', 'log_total_sales', 'total_population', 'Median_Family_Income', 'log_total_sales_centered',]]
                       #'Under_5yrs', '5_17yrs', '18_20yrs', '21_24yrs', '25_34yrs', '35_44yrs','45_54yrs','55_59yrs', '60_64yrs',
                       #'65_74yrs', '75_84yrs', '85+yrs', 'female', 'males_per_100_females', 'Median_Household_Income',
                       #'Per_Capita_Income', 
                       # ]]

# changing column names for personal preference 
zip_frame = zip_frame[['zip_code', 'total_sales', 'log_total_sales', 'log_total_sales_centered', 'Number of Stores Per zip',
                       'volume_sold_liters', 'state_profit_per_bottle',
                       'sales_per_store', 'bottles_sold', 'store_population_ratio', 'consumption_per_capita', 'stores_per_area',
                       'median_age', 'total_population', 'Median_Family_Income',]]
                       #'Under_5yrs', '5_17yrs', '18_20yrs', '21_24yrs', '25_34yrs', '35_44yrs','45_54yrs','55_59yrs', '60_64yrs',
                       #'65_74yrs', '75_84yrs', '85+yrs', 'female', 'males_per_100_females', 'Median_Household_Income',
                       #'Per_Capita_Income']]

# viewing the dataframe
zip_frame

zip_frame.to_csv("data/vodka/zip_frame.csv",index=False)

zip_frame = pd.read_csv ("data/vodka/zip_frame.csv")

zip_frame = zip_frame.dropna()
    
zip_frame.describe()

