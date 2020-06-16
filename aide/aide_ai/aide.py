# python 3.6 only

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle
import datetime

import os


#naturalLanguageProcessing

with open("aide_ai\intents.json") as file:
    data = json.load(file)

with open("aide_ai\data.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)

with open("aide_ai\intents_hindi.json") as file:
    data_hindi = json.load(file)

with open("aide_ai\data_hindi.pickle", "rb") as f:
    words_hindi, labels_hindi, training_hindi, output_hindi = pickle.load(f)

#neural net for english
tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 80)
net = tflearn.fully_connected(net, 80)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.load("aide_ai\model.tflearn")

#neural net for hindi
tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training_hindi[0])])
net = tflearn.fully_connected(net, 80)
net = tflearn.fully_connected(net, 80)
net = tflearn.fully_connected(net, len(output_hindi[0]), activation="softmax")
net = tflearn.regression(net)

model_hindi = tflearn.DNN(net)
model_hindi.load("aide_ai\model_hindi.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    
    s_words = [stemmer.stem(w.lower()) for w in s_words if w != "?" and w != "!"]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

def chat(asr_obj,lang_choice):
    #print("Start talking with the bot (type quit to stop)!")
    #while True:
    #inp = input("You: ")
    inp = asr_obj

    if lang_choice=='false':
        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        print("tag :",tag)
        print("results_index : ",results_index)

        if results[results_index] > 0.7:
            if tag=='ctime' :
                now = datetime.datetime.now()
                return(now.strftime("%H:%M:%S"))
            else:
                for tg in data["intents"]:
                    if tg['tag'] == tag:
                        responses = tg['responses']

                text_output = random.choice(responses) # Text object to be sent for speech synthesis
            #   print(text_output)
                return(text_output)
        else:
    #        print("Sorry, I didn't get that.")
            return("Sorry, I did not get that.")
        
    elif lang_choice=='true':
        results = model_hindi.predict([bag_of_words(inp, words_hindi)])[0]
#    if inp.lower() == "quit":
#       break
        results_index = numpy.argmax(results)
        tag = labels_hindi[results_index]
        print("tag :",tag)
        print("results_index : ",results_index)

        if results[results_index] > 0.7:
            if tag=='ctime' :
                now = datetime.datetime.now()
                return(now.strftime("%H:%M:%S"))
            else:
                for tg in data_hindi["intents"]:
                    if tg['tag'] == tag:
                        responses = tg['responses']

                text_output = random.choice(responses) # Text object to be sent for speech synthesis
            #   print(text_output)
                return(text_output)
        else:
    #        print("Sorry, I didn't get that.")
            return("maaf keejiye, mujhe samajh nahi aya.")

            