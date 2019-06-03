from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	
	print("In predict .............")
	tcc_model = open('model_MNB.pkl','rb')
	print("Model pickle opened .......")
	clf = joblib.load(tcc_model)
	print("Model pickle LOADed .......")

	tcc_vect = open('vectorizer.pkl','rb')
	print("Vectorizer pickle opened .......")
	vectz = joblib.load(tcc_vect)
	print("Vectorizer pickle LOADed .......")


	if request.method == 'POST':
		message = request.form['message']
		#message must contain our comment in string format
		data = [message]
		#test_str_dtm = vectz.transform(data).toarray()
		test_str_dtm = vectz.transform(data)
		my_prediction = clf.predict(test_str_dtm).toarray().tolist()		
	return render_template('result.html',prediction = my_prediction[0])



if __name__ == '__main__':
	app.run(debug=True)
