#####################################################################
#				ML/NLP/NLU Challenge								#
#					Skylar Labs										#
#####################################################################

Hi folks, this is a small ML/NLP/NLU challenge to see if you've got 
what we need at Skylar Labs as a potential candidates for a position
as ML/NLP/NLU Software Engineer.

##Task

The task is simple, build a Retrieval System to pick the right answer
to a certain question, from a dataset.

An user ask a question, your model, process the question and pick the
most suitable answer from a knowledge base dataset, in which the right
answer is.

The user input can be slightly(or even totally different, that can be
cover later on) different from the based question

Example:  
     In dataset:  
         Q: what's your address  
         A: I don't have any address  
     User input:  
        Q: where are you located  
        A: I don't have any address

##Data

You will be provided 2 files to work with:
	
*  train_dataset.txt, is the dataset for training, it's formatted in a way to be easy to process:  
		1- first line is a question  
		2- second line is the answer to the question  
		
   you can take it another way:  
		1- Odd line number is a question  
		2- Even line number is an answer  
		
*  test_dataset.txt is the dataset for testing, it's formatted in a way to be easy to process:  
		1- first line is a question  
		2- second line is the right answer to the question  
   you can take it another way:  
		1- Odd line number is a question  
		2- Even line number is a right answer  
   You can use the test file to benchmarck your system, by comparing the answer ouput by your system and the answer expected  
   
   That is useful to benchmarck your system, and tailor it


##Expectation


What is expected is to get the highest accuracy possible, in the near 
future, the system will evolve in memory awared system, where the 
retrieving will be based on previous answers and questions, so give 
your all, cause you're about to build one of your best and sweetest 
piece of work.


Good luck fellas, may the force be with you !

