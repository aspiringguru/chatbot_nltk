#https://medium.com/analytics-vidhya/building-a-simple-chatbot-in-python-using-nltk-7c8c8215ac6e
#https://github.com/parulnith/Building-a-Simple-Chatbot-in-Python-using-NLTK

# coding: utf-8

# # Meet Robo: your friend
import sys
import nltk
import warnings
warnings.filterwarnings("ignore")

# nltk.download() # for downloading packages (first time only)

import numpy as np
import random
import string # to process standard python strings

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from flask import Flask, render_template, request

app = Flask(__name__)

f=open('chatbot.txt','r',errors = 'ignore')
raw=f.read()
raw=raw.lower()# converts to lowercase

#nltk.download('punkt') # first-time use only
#nltk.download('wordnet') # first-time use only

sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences
word_tokens = nltk.word_tokenize(raw)# converts to list of words

print ("sent_tokens:", sent_tokens[:2])
print ("word_tokens:", word_tokens[:5])

lemmer = nltk.stem.WordNetLemmatizer()
#WordNet is a semantically-oriented dictionary of English included in NLTK.
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
print ("remove_punct_dict:", remove_punct_dict)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "wassup","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    print("function greeting : sentence: ", sentence)
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


def response(user_response):
    print("function response: user_response:", user_response)
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response

@app.route("/")
def home():
    print("app.route - / (home)")
    return render_template("chatterbot.html")

@app.route("/get")
def get_bot_response():
    print("app.route - get (get_bot_response)", file=sys.stdout)
    userText = request.args.get('msg')
    print("app.route - get (get_bot_response): userText:", file=sys.stdout)
    print(userText, file=sys.stdout)
    user_response=userText.lower()
    if(user_response=='thanks' or user_response=='thank you' ):
        print("user_response=='thanks' or 'thank you'", file=sys.stdout)
        return str("You are welcome..")
    else:
        if(greeting(user_response)!=None):
            print("greeting(user_response)!=None", file=sys.stdout)
            return str(greeting(user_response))
        else:
            print("greeting(user_response)==None, remove user_response from sent_tokens", file=sys.stdout)
            temp = response(user_response)
            sent_tokens.remove(user_response)
            return str(temp)
            #return str("I don't understand. Please ask me another question.")


if __name__ == "__main__":
    #print('This is error output', file=sys.stderr)
    print('This is standard output', file=sys.stdout)
    app.run(debug=True, port=80)
