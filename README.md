# FakeNewsDetector
This project is a Flask web application that uses a machine learning model to classify news articles as fake or real. 
The model is trained on the Kaggle Fake and Real News dataset and uses TF-IDF features with Logistic Regression for prediction.

## How to Run
1. Install dependencies:
   pip install -r requirements.txt

2. Train the model (only needed once):
   python train_model.py

3. Start the web app:
   python app.py

4. Open in your browser:
   http://127.0.0.1:5000/

## Files
- train_model.py — trains and saves the model
- app.py — Flask web interface for predictions
- model/ — saved model and vectorizer
- templates/ — HTML interface
- static/ — CSS styling

FakeNewDetectorVideo
https://drive.google.com/file/d/1AGitQJ9jyYN0-MpGPn-CkMahfq4zheoC/view?usp=drive_link
