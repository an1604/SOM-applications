# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from minisom import MiniSom
import re
import nltk

# Download the stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load the dataset
dataset = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'text'], quoting=3)

# Cleaning the text
corpus = []
ps = PorterStemmer()
for i in range(len(dataset['text'])):
    text = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in set(stopwords.words('english'))]
    text = ' '.join(text)
    corpus.append(text)

# Encode the text into numerical data
count = CountVectorizer()
bag_of_words = count.fit_transform(corpus).toarray()
y = pd.get_dummies(dataset['label'], drop_first=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(bag_of_words, y, test_size=0.2, random_state=42)

# Feature scaling
sc = MinMaxScaler(feature_range=(0, 1))
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying the SOM model
som = MiniSom(x=10, y=10, input_len=X_train.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(X_train)
som.train_random(data=X_train, num_iteration=100)

# Visualization
from pylab import bone, pcolor, colorbar, show, plot
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']  # Marker styles for plotting
colors = ['r', 'g']   # Marker colors for plotting

for i, x in enumerate(X_train):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y_train.values[i][0]],  # Access the target labels correctly
         markeredgecolor=colors[y_train.values[i][0]],
         markerfacecolor='None',
         markersize=10,
         markeredgewidth=2)

show()
mappings = som.win_map(data=X_train)
spams = np.concatenate((mappings[8,2], mappings[2,6] , mappings[6,4]))
spams = sc.inverse_transform(spams)
spams = count.inverse_transform(spams)
