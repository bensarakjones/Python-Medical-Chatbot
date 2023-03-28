import random
import json
import pickle
import tkinter as tk
from tkinter import messagebox
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.python.keras.models import load_model
 
import numpy as np

lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())
 
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Display terms and conditions messagebox
response = messagebox.askquestion("DocBot Terms & Conditions", "The DocBot medical chatbot app is provided for informational purposes only and is not intended to be a substitute for professional medical advice, diagnosis, or treatment. The information provided by the DocBot medical chatbot app should not be used for diagnosing or treating a health problem or disease. Always seek the advice of a qualified healthcare provider with any questions you may have regarding a medical condition. The DocBot medical chatbot app is not designed to be used in emergency situations. If you are experiencing a medical emergency, please call your local emergency services or go to the nearest emergency room. The DocBot medical chatbot app does not guarantee the accuracy, completeness, or usefulness of any information provided. The information provided by the DocBot medical chatbot app is subject to change without notice. The DocBot medical chatbot app is not responsible for any actions taken based on the information provided. The user assumes all risks associated with the use of the DocBot medical chatbot app. By using the DocBot medical chatbot app, the user agrees to indemnify and hold harmless DocBot, its affiliates, and their respective officers, directors, employees, agents, licensors, and suppliers from and against any claims, actions, liabilities, losses, damages, and expenses (including reasonable legal fees) arising out of or in connection with the users use of the DocBot medical chatbot app. DocBot reserves the right to modify these terms and conditions at any time without notice. The users continued use of the DocBot medical chatbot app following such modifications constitutes acceptance of the modified terms and conditions. These terms and conditions shall be governed by and construed in accordance with the laws of the jurisdiction in which the DocBot medical chatbot app is provided, without giving effect to any principles of conflicts of law. If any provision of these terms and conditions is held to be invalid or unenforceable, the remaining provisions shall remain in full force and effect. Do you accept the terms and conditions?")

if response == 'no':
    exit()  # Close the app if the user does not accept
 
 
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
 

root = tk.Tk()
root.title("DocBot")

conversation = tk.Text(root, bg="white", fg="black", width=80, height=30)
conversation.pack()

message = tk.StringVar()

message_entry = tk.Entry(root, textvariable=message)
message_entry.pack()

def send_message():
    user_message = message.get()
    ints = predict_class(user_message)
    res = get_response(ints, intents)
    conversation.insert("end", "You: " + user_message + "\n")
    conversation.insert("end", "Chatbot: " + res + "\n")
    message.set("")

send_button = tk.Button(root, text="Send", command=send_message)
send_button.pack()

root.mainloop()

if __name__ == '__main__':
    print("Bot is Running")
    send_message()
        
 
