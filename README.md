#####################################################################
#				ML/NLP/NLU Challenge								#
#														#
#####################################################################

This is an implementation of code for the Hyphen AI Challenge dataset.

##Task

The task was to build a Retrieval System to pick the right answer
to a certain question, from a dataset.

Users ask a question, the model, processes the question and picks the
most suitable answer from a knowledge base dataset.

Example:  
     In dataset:  
         Q: what's your address  
         A: I don't have any address  
     User input:  
        Q: where are you located  
        A: I don't have any address

##Requirements

The requirements are mentioned in the requirement.txt file. To install all dependencies

```
pip install -r requirements.txt
```

##Method Used
The Cleaned Question text is converted into a bag of words (BOW) representation using CountVectorizer. Keras' Deep Neural Network Model is used to link them to the Answers. The model accomodates for changes in input text provided the context is not lost. 'Adam' optimizer is used with Categorical Crossentropy as the Loss function.

##Code Structure
Training File : train_model.py
Eval file: initiate_hyphen_bot.py

PEP8 coding conventions have been followed

| Python File | Pylint Score|
| --- | --- |
| train_model.py | 9.87/10 |


##Running the program

####Step 1: Train the model and get model.h5, vectorizer.pkl and answer.pkl files ( These have already been stored in the repository but it is highly recommeded to run the model again )
```
python train_model.py
```

####Step 2: Run the initiate_hyphen_bot.py file in either the test mode or the conversation mode. 
#####Test mode:
Runs through the entire test set and gives predicted answers. 
```
python initiate_hyphen_bot.py --test
```

#####Conversation mode:
You can enter the questions as you like and the bot will predict the answers
'''
python initiate_hyphen_bot.py --conv
'''


