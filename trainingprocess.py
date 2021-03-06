import readfile as rf
import numpy as np
from sklearn import neighbors

X, Y, X_test, Y_test = rf.getdata()

clf = neighbors.KNeighborsClassifier()
clf.fit(X,Y)

percent = clf.score(X_test, Y_test)
print percent

Xtest, lisdir = rf.get_final_test()

h = clf.predict(Xtest)

rf.result(h, lisdir)
