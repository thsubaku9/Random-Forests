import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import subprocess
import os

os.environ['PATH']+=os.pathsep+ 'C:\graphviz\release\bin'

features=pd.read_csv('temps.csv')
print(features.head(6))
print(features.describe())
#graph based plotting of date and temperature is necessary

#one-hot encoding being performed
features=pd.get_dummies(features)

labels = np.array(features['actual'])
features= features.drop('actual', axis = 1)
#required feature types
feature_list = list(features.columns)
features = np.array(features)

#splitting into training and testing valuess
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

#establishment of baseline
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

'''
Steps to make data machine readable
1.One hot encoding->faster computation
2.Split data into features and labels
3.Convert to arrays
4.Split into training/testing data
'''

#baseline establishment

baseline_preds = test_features[:, feature_list.index('average')]
baseline_preds_2 = train_features[:, feature_list.index('average')]
baseline_error=abs(baseline_preds-test_labels)
baseline_error_2=abs(baseline_preds_2-train_labels)
print('Average baseline error(from testing set): ', round(np.mean(baseline_error), 2))
print('Average baseline error(from training set): ', round(np.mean(baseline_error_2), 2))

#Machine Learning Part
rf=RandomForestRegressor(n_estimators=100, random_state=22)
rf.fit(train_features,train_labels)

predictions=rf.predict(test_features)
errors=abs(predictions-test_labels)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

#mean average percentage error
mape=100*(errors/test_labels)
accuracy=100-np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

tree=rf.estimators_[5]
export_graphviz(tree,out_file='tree.dot',feature_names=feature_list,rounded=True,precision=1)
subprocess.check_call(['dot','-Tpng','tree.dot','-o','result.png'])


'''


pydot needs the GraphViz binaries to be installed anyway, so if you've already generated your dot file you might as well just invoke dot directly yourself. For example:

from subprocess import check_call
check_call(['dot','-Tpng','InputFile.dot','-o','OutputFile.png'])

https://graphviz.gitlab.io/_pages/Download/Download_windows.html
'''
