import numpy as np               # use this scientific library for creating & procesing arrays/matrices
import matplotlib.pyplot as plt  # Backend library for plotting
import matplotlib.colors
from matplotlib import style
from numpy import linalg
import cvxopt
import cvxopt.solvers
import pandas as pd
import sys

csv1_name = sys.argv[1]
#csv2_name = sys.argv[2]

#Read the training data
df = pd.read_csv(csv1_name, header=None)
trainingData = (df.iloc[:,1:]).values

labels = (df.iloc[:,0]).values
labels[labels == 0] = -1

#Read the test data
#df = pd.read_csv(csv2_name, header=None)
#testData = (df.iloc[:,:]).values

class SVM(object):
    """
    Support Vector Machine Classifier/Regression
	
    """
	
    def __init__(self, kernel=None, C=None, loss="hinge"):
        self._margin = 0
        #print ("\n *******************Support Vector Machine Initialization*******************")
        
        if C is not None:
            self._C = float(C)
            print("\nC ->", C)
        else:
            self._C = 10000
            
        if kernel is None:
            self._kernel = self.linear_kernel
        else:
            self._kernel = kernel
        print("Kernel selected ->", self._kernel)

    #Input the data to this method to train the SVM
    def fit(self, X, y):
        n_samples, n_features = X.shape
        #print("\n\nNumber of examples in a sample = ",n_samples , ", Number of features = ", n_features)
        self._w = np.zeros(n_features)

        # Initialize the Gram matrix for taking the output from QP solution
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self._kernel(X[i], X[j])
                #print("K[", i,",", j, "] = ", K[i,j])

        # Here we have to solve the convext optimization problem
        # min 1/2 x^T P x + q^T x
        # s.t.
        #  Gx <= h
        #  Ax = b

        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-np.ones(n_samples))
        #q is a vector of ones
        A = cvxopt.matrix(y, (1, n_samples), 'd')
        #print(A.typecode)
        b = cvxopt.matrix(0.0)

        #G & h are required for soft-margin classifier

        if (self._kernel == self.linear_kernel):
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            #G is an identity matrix with −1s as its diagonal
            # so that our greater than is transformed into less than
            h = cvxopt.matrix(np.zeros(n_samples))
            #h is vector of zeros
        else:
            G_std = np.diag(np.ones(n_samples) * -1)
            h_std = np.identity(n_samples)

            G_slack = np.zeros(n_samples)
            h_slack = np.ones(n_samples) * self._C

            G = cvxopt.matrix(np.vstack((G_std, G_slack)))
            h = cvxopt.matrix(np.hstack((h_std, h_slack)))

        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        alpha = np.ravel(solution['x'])

        # Now figure out the Support Vectors i.e yi(xi.w + b) = 1
        # Check whether langrange multiplier has non-zero value
        sv = alpha > 1e-4
        self._alpha = alpha[sv]
        self._Support_Vectors = X[sv]
        self._Support_Vectors_Labels = y[sv]
        
        print ("\n Total number of examples = ", n_samples)
        print ("\n Total number of Support Vectors found = ", len(self._Support_Vectors))
        print("\n\n Support Vectors are: \n", self._Support_Vectors)
        print("\n\n Support Vectors Labels are: \n", self._Support_Vectors_Labels)

        #Now let us define the decision boundary
        #w = Σαi*yi*xi
        if (self._kernel == self.linear_kernel):
            for i in range(len(self._alpha)):
                #print(i, self._alpha[i], self._Support_Vectors_Labels[i], self._Support_Vectors[i])
                self._w += self._alpha[i] * self._Support_Vectors_Labels[i] * self._Support_Vectors[i]
        else:
            self._w = None
        print("\n Weights are : ",self._w)

        #Now we need to find the margin
        #b = yi − wT xi
        ind = np.arange(len(alpha))[sv]
        self._b = y[ind] - np.dot(X[ind], self._w)

    def linear_kernel(self, x1, x2):
        return np.dot(x1, x2)

    def polynomial_kernel(x, y, p=3):
        return (1 + np.dot(x, y)) ** p

    def gaussian_kernel(x, y, sigma=5.0):
        return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

    def plot_separator(self, X, y):
        plt.figure()

        for i in range(len(X)):
            if (y[i] == 1):
                plt.plot(X[i][0], X[i][1], 'ob')
            else:
                plt.plot(X[i][0], X[i][1], 'xr')

        slope = -self._w[0] / self._w[1]
        intercept = -self._b / self._w[1]
        x = np.arange(0, len(self._Support_Vectors))
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("SVM with soft margin")
        plt.axis("tight")

        #plt.plot(x, (x * slope) + intercept, '--k')

        hyp_x_min = -5
        hyp_x_max = 20

        # (w.x+b) = 1
        # positive support vector hyperplane
        psv1 = (-self._w[0]*hyp_x_min - self._b + 1) / self._w[1]
        psv2 = (-self._w[0]*hyp_x_max - self._b + 1) / self._w[1]
        plt.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k-.', linewidth=0.2)

        # (w.x+b) = -1
        # negative support vector hyperplane
        nsv1 = (-self._w[0]*hyp_x_min - self._b - 1) / self._w[1]
        nsv2 = (-self._w[0]*hyp_x_max - self._b - 1) / self._w[1]
        plt.plot([hyp_x_min,hyp_x_max], [nsv1,nsv2], '--k', linewidth=0.2)

        # (w.x+b) = 0
        # discriminant function
        df1 = (-self._w[0]*hyp_x_min - self._b) / self._w[1]
        df2 = (-self._w[0]*hyp_x_max - self._b) / self._w[1]
        plt.plot([hyp_x_min, hyp_x_max],[df1, df2], 'y')

        plt.show()

    def plot_linear_margin(self):
        plt.figure()
        plt.subplot(221)
        colors = {1:'r',-1:'b'}
        marker = 'o'

        #we need to make three lines in total

        # w.x + b = 0,  
        a0 = -4;
        a1 = (-self._w[0] * a0 - self._margin ) / self._w[1]
        b0 = 4;
        b1 = (-self._w[0] * b0 - self._margin ) / self._w[1]
        pl.plot([a0,b0], [a1,b1], "k")

        # w.x + b = 1

        # w.x + b = -1
        #labels
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("SVM with soft margin")
        plt.axis("tight")
        plt.show()
        
#Instantiate the class instance
svm = SVM()
svm.fit(trainingData, labels)
#svm.plot_learning()
svm.plot_separator(trainingData, labels)
