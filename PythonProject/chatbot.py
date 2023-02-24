import random
import json
import pickle
import tkinter as tk
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.python.keras.models import load_model
 
import numpy as np
lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())
 
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')
 
 
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word)
                      for word in sentence_words]
 
    return sentence_words
 
 
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
 
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)
 
 
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
 
    ERROR_THRESHOLD = 0.25
 
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
 
    results.sort(key=lambda x: x[1], reverse=True)
 
    return_list = []
 
    for r in results:
        return_list.append({'intent': classes[r[0]],
                            'probability': str(r[1])})
    return return_list
 
 
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
 
    result = ''
 
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

#create the main window object
root = tk.Tk()
root.title("Medical Chatbot")

#create the text box where the convesation is going to be
conversation = tk.Text(root, bg="white", fg="black", width=50, height=30)
conversation.pack()

message = tk.StringVar()
#Create the text box where the user inputs their sentence
message_entry = tk.Entry(root, textvariable=message)
message_entry.pack()

#the function ran by the send button
def send_message():
    user_message = message.get()
    ints = predict_class(user_message)
    res = get_response(ints, intents)
    conversation.insert("end", "You: " + user_message + "\n")
    conversation.insert("end", "Chatbot: " + res + "\n")
    message.set("")

#creating the button its self
send_button = tk.Button(root, text="Send", command=send_message)
send_button.pack()

#initiate the main loop
root.mainloop()

if name == 'main':
    print("Bot is Running")
    send_message()
