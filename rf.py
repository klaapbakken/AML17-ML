import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

X = np.load('xtrain.npy')
Y = np.load('ytrain.npy')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

forest = RandomForestClassifier(n_estimators=10)
forest.fit(X_train, Y_train)
print(forest.score(X_train, Y_train))
print(forest.score(X_test, Y_test))

#n_features = X_train.shape[1]
#plt.barh(range(n_features), forest.feature_importances_, align='center')
#feature_names = ['unique words', 'correct words', 'first names', 'last names', 'greetings', 'indicators',
#                 'dates', 'words', 'average word length']
#plt.yticks(np.arange(n_features), feature_names)
#plt.show()