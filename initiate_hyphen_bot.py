"""
Program to run hyphen_bot predictions
"""
import os
import time
from train_model import QAModel
from clint.textui import puts, indent, colored
from clint import arguments
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def input_type():
    """
    Takes command line arguments using Clint
    :return: argument 
    """

    args = arguments.Args()
    return str(args.flags[0])


def start_conv(model):
    """
    For conversation Mode
    :param model: Trained model
    :return: None
    """

    with indent(10):
        puts(colored.red("*Welcome to Hyphen AI's Bot,"
                         " I am Nameless and I'll help you with any que"
                         "stions that you might have!*"))
    puts("Type 'exit' to exit conversation mode")
    while 1:
        with indent(4):

            puts(colored.green("Enter your question here :"))
            question = raw_input("    You :")
            question = str(question)
            if question == "exit":
                break
            answer = model.pred(question=question)
            puts(colored.blue("Nameless Bot: ") + str(answer) + "\n")


def test_mode(model):
    """
    For test dataset evaluation
    :param model: Trained model
    :return: None
    """

    with open('test_dataset.txt', 'r') as filep:
        test = filep.read()
    with open('test-data.txt', 'r') as filep:
        test2 = filep.read()
    test = test + "\n" + test2.decode("utf-16")

    test = test.split("\n")
    test_question = [q for i, q in enumerate(test) if i % 2 == 0]
    test_answer = [q for i,q in enumerate(test) if i % 2 != 0]
    test_answer = [t.strip("\r").encode("utf-8") for t in test_answer]
    print test_answer
    with indent(10):
        puts(colored.red("Here is a test conversation between a User and Nameless Bot"))
    with indent(4):
        count = 0
        l = len(test_answer)
        for question, t_answer in zip(test_question,test_answer):
            #puts(colored.green("\nUser : " + str(question)))
            print "..."
            answer = model.pred(str(question))
            #puts(colored.blue("Nameless Bot: " + str(answer)))
        print " The test file has been run through, the test accuracy is {0} / {1}".format(count, l)


def run():
    """
    Main Function
    :return: None
    """

    model = QAModel()
    inp = input_type()

    if inp == "--conv":
        start_conv(model)

    if inp == "--test":
        test_mode(model)

if __name__ == "__main__":
    run()
