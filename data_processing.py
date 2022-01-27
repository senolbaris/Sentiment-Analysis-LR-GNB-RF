import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
import joblib

data = pd.read_csv("IMDB Dataset.csv")
data10k = data[:10000]

data_x = data10k["review"]
data_y = data10k["sentiment"]

# OneHotEncoding
ohe = OneHotEncoder()
data_y = np.array(data_y)
data_y = data_y.reshape(-1, 1)
data_y = ohe.fit_transform(data_y).toarray()


def processing_sentence(data):
	# Stopwords and Lowercase
	data = data.apply(lambda words: " ".join([word.lower() for word in words.split() if word not in stopwords.words("english")]))
	
	# Stemming
	data= data.apply(lambda words: " ".join([PorterStemmer().stem(word) for word in words.split()]))

	return data

cv = CountVectorizer(max_features=1000)

data_x = processing_sentence(data_x)
data_x = cv.fit_transform(data_x).toarray()

joblib.dump(cv, "bag_of_words.pkl")
np.save("data_x", data_x)
np.save("data_y", data_y)




