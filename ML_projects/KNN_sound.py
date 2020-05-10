

# Import libraries needed
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io.wavfile
from os import listdir


# Helper functions 
def readWav( dir ):
    rate, data = scipy.io.wavfile.read(dir, mmap=False)
    return data


# Here we read the sound wav file and padding them to the same size
fives = [ readWav( 'training_five/' + x ) for x in listdir('training_five') if x != '.DS_Store']
sixes = [ readWav( 'training_six/' + x ) for x in listdir('training_six') if x != '.DS_Store']

# Get the maximum length of sound data
max_len = np.max([np.max([len(a) for a in fives]), np.max([len(a) for a in sixes])])
# We pad the different sound files to same length ( the maxmimum length )
fives_sound = np.asarray([np.pad(a, (0, max_len - len(a)), 'constant', constant_values=0) for a in fives])
sixes_sound = np.asarray([np.pad(a, (0, max_len - len(a)), 'constant', constant_values=0) for a in sixes])


## now we get the image data using sklearn data api
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version = 1)
X, y = mnist["data"], mnist["target"]

fives_image = X[y == '5']
sixes_image = X[y == '6']

fives_image = fives_image[np.random.choice(fives_image.shape[0], 25)]
sixes_image = sixes_image[np.random.choice(sixes_image.shape[0], 25)]


# Before we combine the result, the scale for sound data and image data could be very different, to reduce the effect
# of scales different, we normalize each of them before combine together. 
# Here we use sklearn normalize function, default l2 norm

# Here you can also apply different processings about the original data
from sklearn.preprocessing import normalize
fives_sound = normalize(fives_sound)
sixes_sound = normalize(sixes_sound)
fives_image = normalize(fives_image)
sixes_image = normalize(sixes_image)

# combine them using concatenate along second axis ( columns )
# i.e. features are just put next to each other
fives_combined = np.concatenate((fives_sound, fives_image), axis=1)
sixes_combined = np.concatenate((sixes_sound, sixes_image), axis=1)

# create labels for training
fives_labels = [5 for x in fives_combined]
sixes_labels = [6 for x in sixes_combined]

# Create final data
X = np.concatenate((fives_combined, sixes_combined), axis=0)
y = fives_labels + sixes_labels


# Here we do a stratified training testing split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15, stratify=y)


# Now we can fit our data to Knn classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
k = 25
knn_clf = KNeighborsClassifier(n_neighbors = k)
knn_clf.fit(X_train, y_train)
pred = knn_clf.predict(X_test)


from sklearn.metrics import roc_curve
# Using predict_proba we can get a probablity output from KNN and using different threshold, we can get multiple
# pairs of (TPR, FPR)
scores = knn_clf.predict_proba(X_test)[:,0]
# Here we can use the roc_curve metrics to help for calculate TPR and FPR
fpr, tpr, thresholds = roc_curve(y_test, scores, 5)
# we add (0,0) points for plot ROC curve
fpr = np.insert(fpr,0,0)
tpr = np.insert(tpr,0,0)


# Plot ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.show()


# Levente Papp, 1/1/2019
# It seems like we have a perfect classifier for any K value between between K = 1 and K = 23, with an AUC = 1.0. 
# Therefore, my choice of K for this specific example would be K = 1, because our goal is to maximize AUC while 
# minimizing the computational complexity (K).
for k in range(1, 41):
    knn_clf = KNeighborsClassifier(n_neighbors = k)
    knn_clf.fit(X_train, y_train)
    pred = knn_clf.predict(X_test)
    scores = knn_clf.predict_proba(X_test)[:,0]
    # Here we can use the roc_curve metrics to help for calculate TPR and FPR
    fpr, tpr, thresholds = roc_curve(y_test, scores, 5)
    # we add (0,0) points for plot ROC curve
    fpr = np.insert(fpr,0,0)
    tpr = np.insert(tpr,0,0)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.show()
    print("K = " + str(k))


# ## Problem 2

# Due to the size of the file, my training algorithm could not finish running after 2 hours. I was able to test it on
# smaller data sets. Please see my results below.
# I was able to get an accuracy (95%) with a training set of size 10,000. Given more time or computational power,
# using more data on the "best parameters", the model should be able to get over 97% accuracy, but unfortunately,
# my model did not finish running on the complete data set.
from sklearn.model_selection import GridSearchCV

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

param_grid = [{'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5]}]

knn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(knn_clf, param_grid, cv=5, verbose=3, n_jobs=-1)
grid_search.fit(X_train, y_train)


grid_search.best_params_


grid_search.best_score_


from sklearn.metrics import accuracy_score

y_pred = grid_search.predict(X_test)
accuracy_score(y_test, y_pred)





