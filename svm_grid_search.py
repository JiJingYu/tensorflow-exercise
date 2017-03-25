import tableprint as tp
import pandas as pd
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV

iris = datasets.load_iris()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 2, 4], 'gamma':[0.125, 0.25, 0.5 ,1, 2, 4]}
svr = svm.SVC()
clf = GridSearchCV(svr, parameters, n_jobs=-1)
clf.fit(iris.data, iris.target)
cv_result = pd.DataFrame.from_dict(clf.cv_results_)
with open('cv_result.csv','w') as f:
    cv_result.to_csv(f)
