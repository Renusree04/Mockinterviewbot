import speech_recognition as sr
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from random import choice

question_bank  =  [ 
{
        'question' : 'What is a Data Structure?',
        'file' : 'q1.csv',
        'hint' : 'Array is a Data Structure.'
    },
     {
        'question' : 'Define dynamic data structures.',
        'file' : 'q2.csv',
        'hint' : 'Linked list is a dynamic data structure'
    },
     {
        'question' : 'How are the elements of a 2D Array stored in the memory?',
        'file' : 'q3.csv',
        'hint' : 'Think about how it will be easy to retrive the data'
    },
     {
        'question' : 'What is a linked list?',
        'file' : 'q4.csv',
        'hint' : 'It is like a train'
    },
     {
        'question' : 'Why do we need to do an algorithm analysis?',
        'file' : 'q5.csv',
        'hint' : 'Think about the efficiency'
    }
   
]
   

replies = ['Okay...','Fine....','Hm...','I see....']

def speech_to_text():
    r = sr.Recognizer()
    m = sr.Microphone()
    print("A moment of silence, please...")
    with m as source: 
       r.adjust_for_ambient_noise(source)
    print("Answer!")
    with m as source:
       audio = r.record(source,duration=15)
       print("Got it! ")
    try:
            # recognize speech using Google Speech Recognition
        value = r.recognize_google(audio) # Don't print
        user_ans=format(value)
        print("You said",user_ans)
    except sr.UnknownValueError:
        print("Oops! Didn't catch that")#not ans in duration
        user_ans=speech_to_text()
        #user_ans="I don't know"
    
    except sr.RequestError as e:
        print("No internet connection")
        sys.exit()
        
    return user_ans



def train_data(tv, model, file):
    # Read from csv.
    possible_ans = pd.read_csv(file)
    # Split data into train and test
    ans = possible_ans['Ans']
    label = possible_ans["Label"]
    ans_train, ans_test,label_train, label_test = train_test_split(ans,label,test_size = 0.1, shuffle = True)
    # Train data
    features = tv.fit_transform(ans_train)
    model.fit(features,label_train)
    pickle.dump(model,open('result.pkl','wb'))


def check_ans(tv, model, user_ans):
    features_test = tv.transform(pd.Series(user_ans))
    return model.predict(features_test)[-1]


tv = TfidfVectorizer()
model = svm.SVC()
# Initializing
score = 0
count = 0

for question_number in question_bank:
    print(f"\n{question_number['question']}")
    train_data(tv, model, question_number['file'])
    user_ans=speech_to_text()
    result = check_ans(tv, model, user_ans)
    count += 1
    if result == "Right":
        print(f"Bot: {choice(replies)}")
        score+=10
    else:
        print(f"Bot :Hmm..., I'll give you a hint - {question_number['hint']}")
        user_ans =speech_to_text()
        result = check_ans(tv, model, user_ans)
        if result == "Right":
            score += 5
        print(f"Bot: {choice(replies)}")

final_score = (score/(count*10)) * 100
print(f"\nScore: {final_score}\n")
