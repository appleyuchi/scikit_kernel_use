#-*-coding:utf-8-*- 
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import numpy as np  
from sklearn.svm import SVR  
import matplotlib.pyplot as plt
from sklearn.svm import SVR

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler  



def PolynomialSVR(degree, C=3):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scaler', StandardScaler()),
        # ('linearSVC', LinearSVC(C=C))#注意这两句都行
        ('kernelSVC', SVR(kernel='linear', degree=degree, C=C))#注意这两句都行
    ])


###############################################################################  
# Generate sample data  
X = np.sort(5 * np.random.rand(40, 1), axis=0)  #产生40组数据，每组一个数据，axis=0决定按列排列，=1表示行排列  
y = np.sin(X).ravel()   #np.sin()输出的是列，和X对应，ravel表示转换成行  

###############################################################################  
# Add noise to targets  
y[::5] += 3 * (0.5 - np.random.rand(8))  

###############################################################################  
poly_svc = PolynomialSVR(degree=5)
y_lin =poly_svc.fit(X, y).predict(X)  


###############################################################################  
# look at the results  
lw = 2 #line width  
plt.scatter(X, y, color='darkorange', label='data')  
plt.hold('on')  
plt.plot(X,y_lin, color='c', lw=lw, label='linear kernel')  
plt.xlabel('data')  
plt.ylabel('target')  
plt.title('Support Vector Regression')  
plt.legend()  
plt.show()