# %%

# Подключаем зависимости
import matplotlib.pyplot as pt
import pandas as pd
import numpy as np
from skimage.feature import hog
from sklearn import preprocessing

#Указываем путь до файла с данными
csv_file_path = './archive/emnist-letters-train.csv'

# %%
# Отбираем только нужные нам буквы (A,B,C,D,E,F,H,J,L,K)
df = pd.read_csv(csv_file_path)
data = df[df.iloc[:, 0].isin([1,2,3,4,5,6,8,10,11,12])]
# %%
#Полученные данные
print(data)

# %%
#Случайная буква из набора
d = data.iloc[224, 1:].to_numpy()
d.shape = (28, 28)
pt.imshow(255 - d, cmap='gray')
pt.show()

# %%
X = data.iloc[:,1:].to_numpy()
y = data.iloc[:,0].to_numpy()
print(X)
print(y)

# %%
print(X.shape)
from sklearn.svm import LinearSVC
# %%
# Extract the features and labels
features = X
labels = y

# Extract the hog features
list_hog_fd = []
for feature in features:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1))
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')

# Normalize the features
pp = preprocessing.StandardScaler().fit(hog_features)
hog_features = pp.transform(hog_features)

# Create an linear SVM object
clf = LinearSVC()

# Perform the training
clf.fit(hog_features, labels)

# Save the classifier
joblib.dump((clf, pp), "letters.pkl", compress=3)

# %%
from sklearn.metrics import accuracy_score
csv_file_test_path = './archive/emnist-letters-train.csv'
df_test = pd.read_csv(csv_file_test_path)
data_test = df_test[df_test.iloc[:, 0].isin([1,2,3,4,5,6,8,10,11,12])]
X_test = data_test.iloc[:,1:].to_numpy()
y_test = data_test.iloc[:,0].to_numpy()



list_hog_fd = []
for feature in X_test:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1))
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')

# Normalize the features
pp = preprocessing.StandardScaler().fit(hog_features)
hog_features = pp.transform(hog_features)
target_predicted = clf.predict(hog_features)

# %%
print(accuracy_score(y_test,target_predicted))

# %%
import cv2

imeg = cv2.imread("159.jpg")
gray = cv2.cvtColor(imeg, cv2.COLOR_BGR2GRAY)
print(gray)
a = hog(gray, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1))
print(clf.predict(a))