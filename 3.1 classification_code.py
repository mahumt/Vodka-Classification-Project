# K-Nearest Neighbours
# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
knn_y_pred = knn.predict(X_test)
errors_knn = knn_y_pred != Y_test

#printing out various metrics regarding KNN
print("Nb errors%i, error rate=%.2f" % (errors_knn.sum(), errors_knn.sum() / len(svm_y_pred)))
print("Score:%s" % (knn.score(X_train, Y_train)))
print("Classification Report:\n%s" % (classification_report(Y_test, knn_y_pred, labels=np.unique(knn_y_pred))))

CMat = ConfusionMatrix(Y_test, knn_y_pred)
CMat.stats()
mat = confusion_matrix(Y_test, knn_y_pred)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,xticklabels=label, yticklabels=label)
plt.xlabel('true label')
plt.ylabel('predicted label');


#plotting the number of neighbours needed for the KNN algorithm, changing 'cv' changed optimal number
# creating list of K for KNN
Klist = list(range(1,20,2))
# creating list of cv scores
cv_scores = []
# perform 10-fold cross validation
for k in Klist:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = model_selection.cross_val_score(knn, X_train, Y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())
# changing to misclassification error
MSE = [1 - x for x in cv_scores]
plt.figure()
plt.figure(figsize=(15,10))
plt.title('The optimal number of neighbors', fontsize=20, fontweight='bold')
plt.xlabel('Number of Neighbors K', fontsize=15)
plt.ylabel('Misclassification Error', fontsize=15)
plt.style.use('seaborn-white')
plt.plot(Klist, MSE)
plt.show()

# Finding best K
best_k = Klist[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d." % best_k)

          
# CART: Decision Tree Classifier
DTC  = DecisionTreeClassifier()
DTC.fit(X_train, Y_train)
DTC_y_pred = DTC.predict(X_test)
print(DTC.feature_importances_)

errors_DTC = DTC_y_pred != Y_test
print("Nb errors%i, error rate=%.2f" % (errors_DTC.sum(), errors_DTC.sum() / len(svm_y_pred)))
print("Score:%s" % (DTC.score(X_train, Y_train)))
print("Classification Report:\n%s" % ((classification_report(Y_test, DTC_y_pred, labels=np.unique(DTC_y_pred)))))

DTC_CM = ConfusionMatrix(Y_test, DTC_y_pred)
DTC_CM.stats()

mat = confusion_matrix(Y_test, DTC_y_pred)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,xticklabels=label, yticklabels=label)
plt.xlabel('true label')
plt.ylabel('predicted label');

#setting y= feature importance to plot then against labels 'category'
y = DTC.feature_importances_

plt.style.use('seaborn-white') 
col = plot_variables['category'].unique() #[1031200 1031080 1032200 1032080 1031100 1031090 1031110 1032230]
fig, ax = plt.subplots() 
width = 0.4 # the width of the bars 
ind = np.arange(len(y)) # the x locations for the groups
ax.barh(ind, y, width, color='green')
ax.set_yticks(ind+width/10)
ax.set_yticklabels(col, minor=False)
plt.title('Feature importance in Decision Tree Classifier: CART')
plt.xlabel('Relative importance')
plt.ylabel('Features') 
plt.figure(figsize=(5,5))
fig.set_size_inches(6.5, 4.5, forward=True)

# Generating an image of Decision Tree
dot_data = StringIO()
export_graphviz(DTC, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
