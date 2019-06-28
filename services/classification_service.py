import re
import string
import nltk
import sklearn
import numpy as np
import pandas as pd
import csv
import os.path
import json
import sys
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from wikipedia2vec import Wikipedia2Vec
from collections import Counter
ps = nltk.PorterStemmer()
encoder = preprocessing.LabelEncoder()
labels = './labelPickle.p'
filePickle = './modelGBM.p'
pathJson = './machine learning data/docs/'
mapper = pd.read_csv('./machine learning data/document_departments.csv')
stopword = nltk.corpus.stopwords.words('english')
# pretrained model with 100dim. wiki vectors
wiki2vec = Wikipedia2Vec.load('./enwiki_20180420_100d.pkl')

def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopword]
    return text

def vectorize(inp):
    text = clean_text(inp)
    word_freqs = Counter(text)
    text_vec = np.zeros(100)
    for word in text:
        try: 
            vector = wiki2vec.get_word_vector(word)
            weight = word_freqs[word]/len(text)
            text_vec += weight*vector
        except:
            pass
    return text_vec

class ClassificationService(object):
    """
    service function for predicting department 
    """

    def predict(self, jd):
        """
        TASKS: write the logic here for running the 
        training model on the input text and returning
        the predicted department 
        raise appropriate errors wherever necessary
        """
        model = pickle.load(open(filePickle, 'rb'))
        labelDict = pd.read_csv(labels)
        vectorizedInput = vectorize(jd).reshape(1, -1)
        # output = 'No Department'
        prediction = model.predict(vectorizedInput)
        try:
            for i, x in labelDict.iterrows():
                x = x['labels']
                if int(x) ==  int(prediction[0]):
                    output = str(labelDict[['jobs']].iloc[[prediction[0]]])
        except:
            output = "Model Failed"
        return output


    def train(self):
        """
        - add function params as required
        - train the model and save appropriately
        """
        # print(self)
        jobs = pd.DataFrame()
        listFiles = [pos_json for pos_json in os.listdir(pathJson) if pos_json.endswith('.json')]    
        dataset = pd.concat([pd.read_json(pathJson + x, lines = True) for x in listFiles], ignore_index = True)
        jobs['jd'] = dataset['jd_information'].apply(lambda x: x['description'])
        jobs['Document ID'] = dataset['_id']
        jobs = pd.merge(jobs, mapper, on = 'Document ID')
        jobs['jd'].replace('', np.nan, inplace = True)
        jobs = jobs.dropna()
        X = pd.DataFrame(jobs['jd'])
        y = jobs['Department']
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)
        x_train = pd.DataFrame(x_train.reset_index(drop = True))
        x_test = pd.DataFrame(x_test.reset_index(drop = True))
        # target sets
        encoder.fit(y)
        totalY = encoder.transform(y)        
        # training
        encoder.fit(y_train)
        y_tr = encoder.transform(y_train)
        # test
        encoder.fit(y_test)
        y_te = encoder.transform(y_test)
        # for input sets
        totalX = []
        for x in X['jd']:
            totalX.append(vectorize(x))
        # for training set
        xtr = []
        for x in x_train['jd']:
            xtr.append(vectorize(x))
        # for test set
        xte = []
        for x in x_test['jd']:
            xte.append(vectorize(x))
        # creating a dictionary of labels
        keys = np.unique(np.array(y))
        values = np.unique(np.array(totalY))
        labelDict = dict(zip(keys, values.T))
        # after checking out different learning rates I concluded that this is relatively more precise for test set
        learning_rate = 6
        gb1 = GradientBoostingClassifier(n_estimators=30, learning_rate = learning_rate, max_features=10, max_depth = 10, random_state = 0)
        gb1.fit(xtr, y_tr)
        accuracies = pd.DataFrame(columns = ['learningRate', 'accuracyScore(Training)', 'accuracyScore(Validation)'])
        # print("Learning rate: ", learning_rate)
        # print("Accuracy score (training): {0:.3f}".format(gb1.score(xtr, y_tr)))
        # print("Accuracy score (validation): {0:.3f}".format(gb1.score(xte, y_te)))
        # print()
        accuracies.loc[0] = [learning_rate, gb1.score(xtr, y_tr), gb1.score(xte, y_te)]
        # there is one more learning rate that has impressive accuracy in training set but not on test
        learning_rate = 0.25
        gb2 = GradientBoostingClassifier(n_estimators=30, learning_rate = learning_rate, max_features=10, max_depth = 10, random_state = 0)
        gb2.fit(xtr, y_tr)
        # print("Learning rate: ", learning_rate)
        # print("Accuracy score (training): {0:.3f}".format(gb2.score(xtr, y_tr)))
        # print("Accuracy score (validation): {0:.3f}".format(gb2.score(xte, y_te)))
        # print()
        accuracies.loc[1] = [learning_rate, gb2.score(xtr, y_tr), gb2.score(xte, y_te)]
        # creating pickle for all the input data\
        totalGb = gb1.fit(totalX, totalY)
        # in the pickle we choose learning rate 6 
        w = csv.writer(open(labels, 'w'))
        w.writerow(["jobs", "labels"])
        for key, val in labelDict.items():
            w.writerow([key, val])
        writer = open(filePickle, 'wb')
        try:
            pickle.dump(totalGb, writer)
            writer.close()
        except:
            writer.close()
            pass
        response = accuracies.to_string()
        return response
