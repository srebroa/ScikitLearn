'''
SVM - Support Vector Machines

SVM is widely used in robotics, especially in computer vision for classifying objects and also for classifying various kinds of sensor data in robots.

Advantages of SVM:
- Effective in high dimensional spaces 
- Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient
- Perform better for small datasets

Disadvantages of SVM:
- SVMs do not directly provide probability estimates
- If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions
- It does not do well if the dataset is very large
- It will not be suitable if the dataset has noisy data
'''

# Sensor data classification example
from sklearn import svm
import numpy as np

X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]]) # input (predictor)
y = np.array([1, 2, 3, 4]) # output

model = svm.SVC(kernel='linear',C=1,gamma=1) # create an instance of SVM Classification (SVC) object

model.fit(X,y) # feeding X and y to the SVC object for training the model

# After model training predict the output y value for the given input X
print(model.predict([[1.7,1]])) # labeled as 4
print(model.predict([[0.6,1]])) # labeled as 3
print(model.predict([[-1.9,-1]])) #labeled as 2
