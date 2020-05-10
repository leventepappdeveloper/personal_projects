# Levente Papp
# Homework 1, Problem 1
from PIL import Image
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# This is the helper function that helps us manipulate principal components
def imgCompress(input_mat, numSV=3):
    U,Sigma,VT = np.linalg.svd(input_mat)
    SigRecon = np.zeros((numSV, numSV))
    for k in range(numSV):
        SigRecon[k,k] = Sigma[k]
    reconMat = np.dot(np.dot(U[:,:numSV], SigRecon), VT[:numSV,:])
    return reconMat


# This function computes K-nearest neighbors function with varying numbers of principal components (N)
def tester(K, N):
    ones = []
    zeroes = []
    test = []

    # load training data
    for i in range(40):
        one = Image.open('training_ones/' + str(i) + '.png').convert('L')
        ones.append(np.asarray(one))
        zero = Image.open('training_zeroes/' + str(i) + '.png').convert('L')
        zeroes.append(np.asarray(zero))

    ones_label = [1 for x in ones]
    zeroes_label = [0 for x in zeroes]
    test_label = [0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0]

    # load testing data
    for i in range(20):
        test_data = Image.open('test/' + str(i) + '.png').convert('L')
        test.append(np.asarray(test_data))

    ones = [imgCompress(x, N) for x in ones]
    zeroes = [imgCompress(x, N) for x in zeroes]
    test = [imgCompress(x, N) for x in test]

    ones = [x.flatten() for x in ones]
    zeroes = [x.flatten() for x in zeroes]
    training_x = ones + zeroes
    training_label = ones_label + zeroes_label

    # TRAIN MODEL
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(training_x, training_label)

    test = [x.flatten() for x in test]
    pred = knn.predict(test)

    match = 0
    for j in range(len(test_label)):
        if test_label[j] == pred[j]:
            match = match + 1
    print("Test Accuracy: " + str(match / len(test_label) * 100) + str("%"))


"""Part (b); K = 1
As we can see, a maximum accuracy of 100% is achieved at N = 1, therefore, N = 1 is the smallest number of principal
components so that the method still works properly. """
for n in [5, 4, 3, 2, 1, 0]:
    print("K = ", str(1))
    print("N = ", str(n))
    tester(K=1, N=n)
    print("\n")

"""Part (c); K = 2
Similarly to K=1, a maximum accuracy of 100% is achieved at N = 1, therefore, N = 1 is the smallest number of 
principal components so that the method still works properly. Thus, no, I cannot have a smaller number N of 
principal components even if I increase K from K=1 to K=2. """
for n in [5, 4, 3, 2, 1, 0]:
    print("K = ", str(1))
    print("N = ", str(n))
    tester(K=2, N=n)
    print("\n")

"""Part (c); K = 3
A maximum accuracy of 95% is achieved at N = 1, therefore, N = 1 is the smallest number of 
principal components so that the method still works properly. Thus, no, I cannot have a smaller number N of 
principal components even if I increase K from K=1 to K=3. """
for n in [5, 4, 3, 2, 1, 0]:
    print("K = ", str(1))
    print("N = ", str(n))
    tester(K=3, N=n)
    print("\n")

"""Part (d) My choice would be K=1 and N=1 because as we can see from the results above, this combination maximizes
accuracy while minimizing the number of principal components beings used."""