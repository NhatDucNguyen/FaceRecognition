# import numpy as np

# data = np.load('img.npz')
# lst = data.files

# for item in lst:
#     print(item)
#     print(np.shape(data[item]))


# import numpy as np

# data = np.load('img.npz')
# for key, value in data.items():
#     np.savetxt("somepath" + key + ".csv", value)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC,SVC
from mlxtend.plotting import plot_decision_regions
from matplotlib import pyplot as plt


data = np.load('img.npz')
X= data['arr_0']
y=data['arr_1']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state = 0)
linear = SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(X_train, y_train)
rbf = SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo').fit(X_train, y_train)
poly = SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo').fit(X_train, y_train)
sig = SVC(kernel='sigmoid', C=1, decision_function_shape='ovo').fit(X_train, y_train)
#stepsize in the mesh, it alters the accuracy of the plotprint
#to better understand it, just play with the value, change it and print it
h = .01
#create the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

# create the title that will be shown on the plot
titles = ['Linear kernel','RBF kernel','Polynomial kernel','Sigmoid kernel']
for i, clf in enumerate((linear, rbf, poly, sig)):
    #defines how many plots: 2 rows, 2columns=> leading to 4 plots
    plt.subplot(2, 2, i + 1) #i+1 is the index
    #space between plots
    plt.subplots_adjust(wspace=0.4, hspace=0.4) 
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.PuBuGn, alpha=0.7)
    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], cmap=plt.cm.PuBuGn, edgecolors='grey')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])
    plt.show()
    
# a_byte_array = bytearray(r)
# byte_list = []

# for byte in a_byte_array:
#     binary_representation = bin(byte)
#     byte_list.append(binary_representation)
# y=np.array(byte_list)

# Training a classifier

# Plot data points and color using their class
# color = ['black' if c == 0 else 'blue' for c in y]
# plt.scatter(X[:,0], X[:,1], c=color)

# # Create the hyperplane
# w = svm.coef_[0]
# a = -w[0] / w[1]
# xx = np.linspace(-2.5, 2.5)
# yy = a * xx - (svm.intercept_[0]) / w[1]

# # Plot the hyperplane
# plt.plot(xx, yy)
# plt.axis("off"), plt.show()

