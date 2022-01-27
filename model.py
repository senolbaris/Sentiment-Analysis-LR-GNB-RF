import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import warnings
warnings.filterwarnings("ignore")

data_x = np.load("data_x.npy")
data_y = np.load("data_y.npy")
data_y = np.delete(data_y, 1, 1)

cv = joblib.load("bag_of_words.pkl")

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=1)

# Logistic Regression
lg = LogisticRegression()
lg.fit(x_train, y_train.ravel())
y_pred_lg = lg.predict(x_test)
lg_accuracy = accuracy_score(y_test, y_pred_lg)
print("Logistic Regression: ", lg_accuracy) 

# Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(x_train, y_train.ravel())
y_pred_gnb = gnb.predict(x_test)
gnb_accuracy = accuracy_score(y_test, y_pred_gnb)
print("Gaussian Naive Nayes: ", gnb_accuracy)

# Random Forest
rf = RandomForestClassifier()
rf.fit(x_train, y_train.ravel())
y_pred_rf = rf.predict(x_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print("Random Forest: ", rf_accuracy)


