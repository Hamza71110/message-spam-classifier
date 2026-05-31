import streamlit as st
import nltk
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def preprocessing(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfid = pickle.load(open('vectorixer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Message Spam Classifier")

input_sms = st.text_area("Enter your message here")

if st.button('Press to Predict'):

    # 1. preprocess
    transformed_sms = preprocessing(input_sms)
    # 2. vectorize
    vector_input = tfid.transform([transformed_sms]).toarray()
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Provided Message is Spam")
    else:
        st.header("Provided Message is not Spam")