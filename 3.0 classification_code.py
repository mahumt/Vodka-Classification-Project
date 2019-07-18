import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.graph_objs import *
from plotly.offline import *  #offline version to import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks
cufflinks.go_offline()  #to use cufflinks offline  #https://plot.ly/ipython-notebooks/cufflinks/
import seaborn as sns
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-white') #for jupyter dark, change style
init_notebook_mode()   #to connect jupyter with javascript as plotly is interactive
%config InlineBackend.figure_format = 'svg' #Graphics in SVG format are more sharp and legible
import scipy.sparse
import warnings
import pycountry
import random

from pandas.tools.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from pandas_ml import ConfusionMatrix
from sklearn import preprocessing

# there need to be some slight change to the preprocessed data: to make it usable for classificatio purposes
df1 = pd.read_csv ('data/vodka/lowa_liquor_sales.csv')
zip_frame = pd.read_csv("data/vodka/zip_frame.csv")

log_df= pd.merge(zip_frame,df1[['category']], left_on='zip_code', how='right', left_index=True, right_index=True)
print(log_df.shape)
log_df.head()
log_df.to_csv("data/vodka/log_df.csv", index=False, sep='|')

#Once the changes are done and the file is saves as a 'csv' file, it can be called forward without the any piror code.
log_df=pd.read_csv("data/vodka/log_df.csv" , sep= "|" , low_memory=False)
print(log_df.shape)
log_df.head()

# slight preprocessing and exploration of the data is reuired for feature selection and such:
# to find if there are null values
log_df.isnull().sum() # around 90%+ data has null values (this is going to be problem later)

# to find all the categories of the categorical variable
print (log_df['category'].unique())

# to see which cateogry is the most popular
log_df.groupby('category').count()

log_df.columns

#for feature selection (i.e. to select variales that will be X_variables) some sort of test is required
# the easiest way to find out which variables should be used is to draw a heatmap that charts the correlation:
#the highest amount of correlation is found in various different popultion variables, sales variables  and age variables.
#all of these variables are dropped and we only take total sales and total population variables.

X_heatmap = log_df[['zip_code', 'total_sales', 'log_total_sales',
       'Number of Stores Per zip', 'volume_sold_liters',
       'state_profit_per_bottle', 'sales_per_store', 'bottles_sold',
       'store_population_ratio', 'consumption_per_capita', 'stores_per_area',
       'median_age', 'total_population', 'Median_Family_Income', 'Under_ 5yrs',
       '5_17yrs', '18_20yrs', '21_24yrs', '25_34yrs', '35_44yrs', '45_54yrs',
       '55_59yrs', '60_64yrs', '65_74yrs', '75_84yrs', '85+yrs', 'female',
       'males_ per_ 100_females', 'Median_Household_Income',
       'Per_Capita_Income', 'category' ]]
corr1 = X_heatmap.corr()
print (corr1['category'].sort_values(ascending=False)[:15])
f, ax = plt.subplots(figsize=(30, 30))

# Linear Seperability test
# another thing to test is whether the data is linearly seperable i.e. a straight line of hyperplane can cut through all the categories.
# we need to know whther linear classifiers or linear kernels will work efficently.
# there are several ways to test this: linear programming, a perceptron (commonly single layer), 
# clustering with 100% purity: linear classifer 
# with no errors and the most visual and easy to understand: computational geometry using ```ConvexHull```


plot_variables = log_df[['total_sales', 'store_population_ratio', 
                         'consumption_per_capita','stores_per_area',
                         'median_age', 'total_population',
                         'males_ per_ 100_females', 'Per_Capita_Income',
                         'category']]
plot_variables.shape ,plot_variables.columns[0], plot_variables.columns[5], plot_variables.columns[8]
plot_variables.fillna(plot_variables.mean(), inplace=True) 

sns.set_style("darkgrid")
ax = sns.stripplot(x='total_sales', y='total_population', hue='category', data=plot_variables, jitter=False, palette="Set2", dodge=True)

from scipy.spatial import ConvexHull
plt.style.use('seaborn-white') 
plt.clf()
plt.figure(figsize=(10, 6))
#plt.scatter(log_df[['category']])
names = log_df['category'].unique() # target category names
label =  log_df['category'].astype(np.int)
colors = ['blue','red','green','yellow','violet', 'indigo', 'orange', 'black']
plt.title('Total sales and total population by categories of vodka')
plt.xlabel(plot_variables.columns[0]) # predictor column number 0= total sales
plt.ylabel(plot_variables.columns[5])
for i in range(len(names)):
    bucket = plot_variables[plot_variables.iloc[:,[8]] == i]
    bucket = plot_variables.iloc[:,[0,5]].values
    hull = ConvexHull(bucket)
    plt.scatter(bucket[:, 0], bucket[:, 1], label=names[i]) 
    for j in hull.simplices:
        plt.plot(bucket[j,0], bucket[j,1], colors[i])
plt.legend()
plt.show();
sns.heatmap(corr1, vmax=.8, annot_kws={'size': 10}, annot=True);

# Choosing X varaibles (predictors) and Y variable (target):
predictors = log_df[['total_sales', 'store_population_ratio', 
                     'consumption_per_capita','stores_per_area',
                     'median_age', 'total_population','males_ per_ 100_females', 
                     'Per_Capita_Income']]
print(predictors.shape)

# Filling null values
#filling with 0 messes with train_test_split
predictors.fillna(predictors.mean(), inplace=True) # better to fill predictors, less demadning on RAM
#log_df = log_df.dropna() # dropping null values leaves us with round 1% of the data.

#predictors are off different scale: so we standardize them.
#predictors = preprocessing.normalize(predictors) # normalize the data attributes
predictors = preprocessing.scale(predictors) # standardize the data attributes
predictors.head()
predictors = predictors.sample(200000) #bigger the sample better, depends on how much RAM capacity you have 

label = log_df['category'].unique() # non-encoded variable will be used later for label purposes in plots

#choosing the target variable
target = log_df['category']
target = target.sample(200000)

# label encoder used to change the categories into a proper cateogrical value
le = preprocessing.LabelEncoder() #label encoder for 1-d array, so if dataframe encoded: df.apply(LabelEncoder().fit_transform)
target = le.fit_transform(target)

#split the data in 80% training set and 20% test set
X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size = 0.20)

# Preliminary comparison of all models
#  Algorithms for models
models = []
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('QDA', QuadraticDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('RNDM FRST', RandomForestClassifier()))
models.append(('NB', GaussianNB()))
models.append(('LR', LogisticRegression(multi_class='multinomial',solver ='saga', tol= 1e-2, max_iter=1000)))
models.append(('NL-SVM', SVC()))

warnings.filterwarnings('ignore') # to ignore any warning and get a unbroken lines of outputs
# Test options and evaluation metric
num_instances = len(X_train)
# prepare configuration for cross validation test harness
seed = 7
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed) #(n=num_instances, n_folds=num_folds=n_splits=n-fold CV)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	message = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) #mean*100 and std* to show percentage
	print(message)
  
 # boxplot for algorithm comparison
plt.style.use('seaborn-white') 
fig = plt.figure(figsize=(10, 10))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.style.use('seaborn-white')
plt.show();

# Both QDA and Gaussian Naive Bayes, have wildly different mean and standard deviations
# due to either being unsuitable or being too simple
# hence they are both dropped

#define model_func which fits, trains and predicts using different models: better than running messier repetitive code for each model
from sklearn import model_selection, metrics

def model_func(alg, X_train, X_test, Y_train, Y_test, target, predictors, filename):
    #Fit the algorithm on the data
    algorithm=alg.fit(X_train, Y_train)
    print (algorithm)
    
    #Predict training set:
    dtrain_predictions = alg.predict(X_train)

    #Perform cross-validation:
    cv_score = model_selection.cross_val_score(alg, X_train, Y_train , cv=10, scoring='neg_mean_squared_error') 
	#scoring='accuracy', cv=kfold
    #kfold = model_selection.KFold(n_splits=10, random_state=seed) #(n=num_instances, n_folds=num_folds=n_splits=n-fold CV)
    cv_score = np.sqrt(np.abs(cv_score))
    
    #Print model report:
    print("\nModel Report")
    print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error((Y_train), dtrain_predictions)))
    print("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),
									     np.std(cv_score), 
									     np.min(cv_score),
									     np.max(cv_score)))
    
    #Predict on testing data and computing error rate:
    #dtest[target] = alg.predict(dtest[predictors])
    predictions = alg.predict(X_test)
    errors = predictions !=  Y_test
    number_of_errors = errors.sum()
    error_rate = errors.sum() / len(predictions)
    
    
    #Print R-squared
    sse = ((Y_test - predictions) ** 2).sum(axis=0)
    tse = ((Y_test  - np.average(Y_test , axis=0)) ** 2).sum(axis=0))
    print("R-squared : %.4g" % (1 - (sse / tse)))
    print("RMSE Test : %.4g" % np.sqrt(metrics.mean_squared_error((Y_test), predictions)))
    
    
    print("Score:%s" % (alg.score(X_train, Y_train)))
    print("Decision Function:\n%s" % (alg.decision_function(X_test)))
    print("Intercept:%s" %(alg.intercept_))
    print("Coefficents:\n%s"%(alg.coef_))
    print("Number of errors=%i, error rate=%.2f" % (number_of_errors, error_rate))
    print("Classification Report:\n%s" % (classification_report(Y_test, predictions, labels=np.unique(predictions))))
    
    CMat = ConfusionMatrix(Y_test, predictions)
    print("Statistics regarding classification model =%s" % (CMat.stats()))
    
    # ROC-AUC score: closer to 1 the better. This metric dosen't work in multinomial cases.
    #k_fold = model_selection.KFold(n_splits=10, random_state=7)
    #results = model_selection.cross_val_score(alg, predictors, target, cv = k_fold, scoring='roc_auc')
    #print("AUC: %.3f (%.3f)") % (results.mean(), results.std())
          
    #Plot the confusion matrix
    mat = confusion_matrix(Y_test, RMC_y_pred)
    sns.heatmap(mat, annot=True, fmt='d', cbar=True, xticklabels=label, yticklabels=label, linewidths=.5)
    plt.xlabel('true label')
    plt.ylabel('predicted label');
    
    # Graph on testing data: line/model
    plt.scatter(Y_test, predictions , marker='+', color='r')
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.show()


warnings.filterwarnings('once') # to get warnings once in case the coefficents arent converging or any other problems

# L1 regularization: Ridge, L2 regularization: Lasso

#For multinomial there are only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ that handle multinomial loss.
#only SAGA regularizes has both L1 and L2
# when multi_class='multinomial'  SoftMax function used instead of the common sigmoid
logregl1 = LogisticRegression(multi_class='multinomial',solver ='saga', tol= 1e-2, penalty= 'l1', max_iter=1000)
logregl2 = LogisticRegression(multi_class='multinomial',solver ='saga', tol = 1e-2, penalty= 'l2', max_iter=1000)
logregCVl1 = LogisticRegressionCV(multi_class='multinomial',solver ='saga', tol=1e-2, cv = 3, penalty = 'l1', max_iter=1000)
logregCVl2 = LogisticRegressionCV(multi_class='multinomial',solver ='saga', tol=1e-2, cv=3, penalty = 'l2' ,max_iter=1000)

LDA = LinearDiscriminantAnalysis()

# Linear SVC is not really useful due to linear kernels, but we want to see the effect of L1 and L2 regularizaiotn
svmlinl1 = LinearSVC(penalty='l1', dual=False, max_iter=10000)# by default LinearSVC uses squared_hinge as loss
svmlinl2 = LinearSVC(dual=False, max_iter=10000)
svm = SVC(max_iter=10000, probability=True, kernel = 'rbf') #rbf best due to non-parametric methods and large number of variables


model_func(logregl1, X_train, X_test, Y_train, Y_test, predictors, target, 'data\vodka\alg.csv')
model_func(logregl2, X_train, X_test, Y_train, Y_test, predictors, target, 'data\vodka\alg.csv')
model_func(logregCVl1, X_train, X_test, Y_train, Y_test, predictors, target, 'data\vodka\alg.csv')
model_func(logregCVl2, X_train, X_test, Y_train, Y_test, predictors, target, 'data\vodka\alg.csv')
model_func(LDA, X_train, X_test, Y_train, Y_test, predictors, target, 'data\vodka\alg.csv')
model_func(svmlinl1, X_train, X_test, Y_train, Y_test, predictors, target, 'data\vodka\alg.csv')
model_func(svmlinl2, X_train, X_test, Y_train, Y_test, predictors, target, 'data\vodka\alg.csv')
model_func(svm, X_train, X_test, Y_train, Y_test, predictors, target, 'data\vodka\alg.csv')
