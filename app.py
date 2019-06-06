from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import MultinomialNB
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from sklearn.externals import joblib

import string
import nltk
import re
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer 
from nltk import pos_tag
from nltk.corpus import wordnet

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	
    print("In predict .............")
    tcc_model = open('model.pkl','rb')
    print("Model pickle opened .......")
    clf = joblib.load(tcc_model)
    print("Model pickle LOADed .......")

    tcc_vect = open('tokenizer.pkl','rb')
    print("tokenizer pickle opened .......")
    vectz = joblib.load(tcc_vect)
    print("Vectorizer pickle LOADed .......")
    if request.method == 'POST':
        message = request.form['message']
        message = clean_text(message)
        data = [message]
        test_sequences = vectz.texts_to_sequences(data)
        x_test = pad_sequences(test_sequences,maxlen=175)
        ynew = clf.predict(x_test)
        K.clear_session()
    return render_template('result.html',prediction = ynew[0])

    # array([[0.70259744, 0.05300561, 0.0574163 , 0.7864018 , 0.0819792 ,
    # 0.01291738]], dtype=float32)


def clean_text(text):
    
    ## Remove puncuation
    text = text.translate(string.punctuation)
    
    ## Convert words to lower case and split them
    text = text.lower().split()
    
    ## Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if  (not w in stops and len(w) >= 3 and w)]
    
    text = " ".join(text)
    ## Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r'[0-9\.]+', '', text)
    ## Stemming
    text = text.split()
    lemmatizer = WordNetLemmatizer() 
    lem_words = [lemmatizer.lemmatize(word) for word in text]
    lem_words = [lemmatizer.lemmatize(word,pos="v") for word in lem_words]
    lem_words = [lemmatizer.lemmatize(word,pos="a") for word in lem_words]
    text = " ".join(lem_words)
    return text

if __name__ == '__main__':
	app.run(debug=True)
