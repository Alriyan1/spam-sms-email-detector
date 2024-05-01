import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title('Email/SMS spam classifier')


def transform_text(text):
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

input_sms=st.text_area('Enter the message')

if st.button('Predict'):
    #preprocess
    transformed_sms=transform_text(input_sms)
    #vectorise
    vector_input=tfidf.transform([transformed_sms])
    #predict
    result=model.predict(vector_input)[0]
    #display
    if result==1:
        st.header('Spam')
    else:
        st.header('Not Spam')

# Oh k...i'm watching here
# Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's
# Even my brother is not like to speak with me. They treat me like aids patent.