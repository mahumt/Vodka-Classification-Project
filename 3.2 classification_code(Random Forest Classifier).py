#Random Forest Classifier
RMC  = RandomForestClassifier(n_estimators=100)
RMC.fit(X_train, Y_train)
RMC_y_pred = RMC.predict(X_test)
errors_RMC = RMC_y_pred != Y_test

print("Nb errors%i, error rate=%.2f" % (errors_RMC.sum(), errors_RMC.sum() / len(errors_RMC)))
print("Score:%s" % (RMC.score(X_train, Y_train)))
print("Feature Importance:\n%s"% (RMC.feature_importances_))
print("Classification Report:\n%s" % (classification_report(Y_test, RMC_y_pred, labels=np.unique(RMC_y_pred))))

CMat = ConfusionMatrix(Y_test, RMC_y_pred)
CMat.stats()

mat = confusion_matrix(Y_test, RMC_y_pred)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,xticklabels=label, yticklabels=label)
plt.xlabel('true label')
plt.ylabel('predicted label');

#setting z = feature importance and plotting them against 'category' label
z = RMC.feature_importances_

plt.style.use('seaborn-white') 
col = plot_variables['category'].unique() #[1031200 1031080 1032200 1032080 1031100 1031090 1031110 1032230]
fig, ax = plt.subplots() 
width = 0.4 # the width of the bars 
ind = np.arange(len(z)) # the x locations for the groups
ax.barh(ind, z, width, color='green')
ax.set_yticks(ind+width/10)
ax.set_yticklabels(col, minor=False)
plt.title('Feature importance in Random Forest Classifier')
plt.xlabel('Relative importance')
plt.ylabel('Features') 
plt.figure(figsize=(5,5))
fig.set_size_inches(6.5, 4.5, forward=True)

# Generating an image of an tree from the random forest
dot_data = StringIO()
tree.export_graphviz(RMC.estimators_[0], out_file='tree_from_forest.dot')
(graph,) = pydot.graph_from_dot_file('tree_from_forest.dot')
graph.write_png('tree_from_forest.png')
from PIL import Image
try:  
    img  = Image.open('tree_from_forest.png')  
except IOError: 
    pass
img
          
#Mapping the Deicsion Function
          
# The decision estimator has an attribute called tree_  which stores the entire tree structure and allows access to low level attributes.
#The binary tree: tree_  represented as number of parallel arrays. The i-th element of each array holds info about the node `i`. Node 0 is the tree's root.
#NOTE:  Some of the arrays only apply to either leaves or split nodes, resp. this case: values of nodes of the other type are arbitrary

# Among those arrays, we have:
#   - left_child, id of the left child of the node and vice versa
#   - 'feature' used for splitting the node
#   - 'threshold' value at the node

# Using those arrays, we can parse the tree structure:

estimator = RandomForestClassifier()
estimator.fit(X_train, Y_train)
n_nodes = estimator.tree_.node_count
children_left = estimator.tree_.children_left
children_right = estimator.tree_.children_right
feature = estimator.tree_.feature
threshold = estimator.tree_.threshold


# The tree structure can be traversed to compute various properties such  as the depth of each node and whether or not it is a leaf.
node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
stack = [(0, -1)]  # seed is the root node id and its parent depth
while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    node_depth[node_id] = parent_depth + 1

    # If we have a test node
    if (children_left[node_id] != children_right[node_id]):
        stack.append((children_left[node_id], parent_depth + 1))
        stack.append((children_right[node_id], parent_depth + 1))
    else:
        is_leaves[node_id] = True

print("The binary tree structure has %s nodes and has "
      "the following tree structure:"
      % n_nodes)
for i in range(n_nodes):
    if is_leaves[i]:
        print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
    else:
        print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
              "node %s."
              % (node_depth[i] * "\t",
                 i,
                 children_left[i],
                 feature[i],
                 threshold[i],
                 children_right[i],
                 ))
print()

# First retrieve the decision path of each sample: allows to retrieve the node indicator functions. 
#A non zero element of # indicator matrix at the position (i, j) indicates that the sample i goes  through the node j.

node_indicator = estimator.decision_path(X_test)

# Similarly, we can also have the leaves ids reached by each sample.

leave_id = estimator.apply(X_test)

# Now, it's possible to get the tests that were used to predict a sample(s). First, let's make it for the sample.

sample_id = 0
node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                    node_indicator.indptr[sample_id + 1]]

print('Rules used to predict sample %s: ' % sample_id)
for node_id in node_index:
    if leave_id[sample_id] == node_id:
        continue

    if (X_test[sample_id, feature[node_id]] <= threshold[node_id]):
        threshold_sign = "<="
    else:
        threshold_sign = ">"

    print("decision id node %s : (X_test[%s, %s] (= %s) %s %s)"
          % (node_id,
             sample_id,
             feature[node_id],
             X_test[sample_id, feature[node_id]],
             threshold_sign,
             threshold[node_id]))

# For a group of samples, we have the following common node.
sample_ids = [0, 1]
common_nodes = (node_indicator.toarray()[sample_ids].sum(axis=0) ==
                len(sample_ids))

common_node_id = np.arange(n_nodes)[common_nodes]

print("\nThe following samples %s share the node %s in the tree"
      % (sample_ids, common_node_id))
print("It is %s %% of all nodes." % (100 * len(common_node_id) / n_nodes,))

#Mappingo out the deicsion path of Random Forest
estimator = RandomForestClassifier(n_estimators=10, random_state=0)
estimator.fit(X_train, Y_train)

n_nodes_ = [t.tree_.node_count for t in estimator.estimators_] #estimator.tree_.node_count
children_left_ = [t.tree_.children_left for t in estimator.estimators_]
children_right_ = [t.tree_.children_right for t in estimator.estimators_]
feature_ = [t.tree_.feature for t in estimator.estimators_]
threshold_ = [t.tree_.threshold for t in estimator.estimators_]

def explore_tree(estimator, n_nodes, children_left,children_right, feature,threshold, suffix='', print_tree= False, sample_id=0, feature_names=None):

    if not feature_names:
        feature_names = feature
    assert len(feature_names) == X_train.shape[1], "The feature names do not match the number of features."
    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1
        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True
    print("The binary tree structure has %s nodes"  % n_nodes)
    if print_tree:
        print("Tree structure: \n")
        for i in range(n_nodes):
            if is_leaves[i]:
                print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
            else:
                print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
                      "node %s." % (node_depth[i] * "\t",i,children_left[i],
                         feature[i], threshold[i], children_right[i],))
            print("\n")
        print()
    # First let's retrieve the decision path of each sample. The decision_path
    # method allows to retrieve the node indicator functions. A non zero element of
    # indicator matrix at the position (i, j) indicates that the sample i goes
    # through the node j.
    node_indicator = estimator.decision_path(X_test)
    # Similarly, we can also have the leaves ids reached by each sample.
    leave_id = estimator.apply(X_test)
    # Now, it's possible to get the tests that were used to predict a sample or
    # a group of samples. First, let's make it for the sample.
    #sample_id = 0
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]: node_indicator.indptr[sample_id + 1]]
    print(X_test[sample_id,:])
    print('Rules used to predict sample %s: ' % sample_id)
    for node_id in node_index:
        # tabulation = " "*node_depth[node_id] #-> makes tabulation of each level of the tree
        tabulation = ""
        if leave_id[sample_id] == node_id:
            print("%s==> Predicted leaf index \n"%(tabulation))
        if (X_test[sample_id, feature[node_id]] <= threshold[node_id]):
            threshold_sign = "<="
        else:
            threshold_sign = ">"
        print("%sdecision id node %s : (X_test[%s, '%s'] (= %s) %s %s)"
              % (tabulation,node_id,sample_id,feature_names[feature[node_id]],  X_test[sample_id, feature[node_id]],threshold_sign,threshold[node_id]))
    print("%sPrediction for sample %d: %s"%(tabulation, sample_id,estimator.predict(X_test)[sample_id]))
    # For a group of samples, we have the following common node.
    sample_ids = [sample_id, 1]
    common_nodes = (node_indicator.toarray()[sample_ids].sum(axis=0) ==len(sample_ids))
    common_node_id = np.arange(n_nodes)[common_nodes]
    print("\nThe following samples %s share the node %s in the tree" % (sample_ids, common_node_id))
    print("It is %s %% of all nodes." % (100 * len(common_node_id) / n_nodes,))
    for sample_id_ in sample_ids:
        print("Prediction for sample %d: %s"%(sample_id_,estimator.predict(X_test)[sample_id_]))

#Implementation of decision path
for i,e in enumerate(estimator.estimators_):
    print("Tree %d\n"%i)
explore_tree(estimator.estimators_[i],n_nodes_[i],children_left_[i], children_right_[i],
             feature_[i],threshold_[i], suffix=i, sample_id=1, feature_names = ["Feature_%d"%i for i in range(X_train.shape[1])])
print('\n'*2)
