import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
import numpy as np


def clean_function(corpus):
    """
    Clean Function
    :param corpus: Enter data for clean vector
    :return: Returns clean string vector
    """
    vocab = re.sub("[^a-zA-Z]", " ", corpus)
    vocab = vocab.lower().split()
    vocab = " ".join(vocab)
    return vocab


class QAModel(object):
    """
    Class QAModel
    """
    def __init__(self):
        self.question = None
        self.train_x = []
        self.train_y = []
        self.answer = None
        self.vectorizer = None
        self.vocab = None
        self.model = None

    def load_files(self, training_file):
        """

        :param training_file: Takes in the training file name
        :return: Saves extracted question and answer
        """
        with open(training_file, 'r') as filep:
            train = filep.read()

        self.vocab = train
        train = train.split("\n")
        self.question = [q for i, q in enumerate(train) if i % 2 == 0]
        self.answer = [q for i, q in enumerate(train) if i % 2 != 0]
        joblib.dump(self.answer, 'answer.pkl')

    def define_vocab(self):
        """
        Initializes Vocab function
        :return:
        """
        self.vocab = [clean_function(self.vocab)]

    def get_datasets(self):
        """
        Load datasets into the class object
        :return:
        """
        sample_size = len(self.answer)
        output_vector = np.zeros(sample_size)
        for trainx, i in zip(self.question, range(sample_size)):
            trainx = clean_function(trainx)
            self.train_x.append(trainx)
            output_vector[i] = i
        train_x = self.convert_to_word_vectors(self.vocab, self.train_x)
        self.train_x = train_x
        train_y = to_categorical(output_vector)
        self.train_y = train_y

    def convert_to_word_vectors(self, vocab, train_x):
        """
        :param vocab: vocabulary of all words
        :param train_x: Input String Vector
        :return: Returns Input Numerical Vector
        """
        vectorizer = CountVectorizer(tokenizer=None, analyzer='word')
        vectorizer.fit(vocab)
        train_x = vectorizer.transform(train_x).toarray()
        self.vectorizer = vectorizer
        joblib.dump(vectorizer, 'vectorizer.pkl')
        return train_x

    def model_train(self):
        """
        Trains the Deep Net Model
        :return: 
        """
        model = Sequential()
        model.add(Dense(512, input_shape=[np.shape(self.train_x)[1]], activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(np.shape(self.train_x)[0], activation='softmax'))
        model.summary()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(self.train_x, self.train_y, nb_epoch=1500, batch_size=7)
        self.model = model
        model.save('model.h5')

    def pred(self, question):
        """
        Function to predict the answer from the model
        :param question: Takes question input
        :return: Returns the answer
        """
        question = clean_function(question)
        vectorizer = joblib.load('vectorizer.pkl')
        model = load_model('model.h5')
        ques = vectorizer.transform([question]).toarray()
        ans = model.predict_classes(ques, verbose=0)
        answer = joblib.load('answer.pkl')
        return answer[int(ans)]


def run():
    """
     :return: 
    """
    model = QAModel()
    model.load_files('training_dataset.txt')
    model.define_vocab()
    model.get_datasets()
    model.model_train()

if __name__ == "__main__":
    run()
