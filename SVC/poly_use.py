# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
# @Author: appleyuchi
# @Date:   2018-12-15 20:35:43
# @Last Modified by:   appleyuchi
# @Last Modified time: 2018-12-17 19:13:57
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from plot import *
X, y = datasets.make_moons(noise=0.15, random_state=666)
# plt.scatter(X[y==0, 0], X[y==0, 1])
# plt.scatter(X[y==1, 0], X[y==1, 1])
# plt.show()
print"X=",X
print len(X)
print"y=",y
print len(y)


from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import sklearn.svm
# print dir(sklearn.svm)
def PolynomialSVC(degree, C=1):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scaler', StandardScaler()),
        # ('linearSVC', LinearSVC(C=C))#注意这两句都行
        ('kernelSVC', SVC(kernel='poly', degree=degree, C=C,class_weight={0:40,1:70})
        )#注意这两句都行
    ])

poly_svc = PolynomialSVC(degree=2)
poly_svc.fit(X, y)

plot_decision_boundary(poly_svc, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.show()