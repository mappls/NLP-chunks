from keras.preprocessing.sequence import pad_sequences
from nltk import word_tokenize
import numpy as np
import keras
import json
import os

# Parameters of the model
MAX_SENTENCE_LENGTH = 60
OUTPUT_DIMENSION = 5
DATASET_FILEPATH = "Datasets/ner_dataset.txt"
WORD_INDEX_FILEPATH = "word_index.json"
MODEL_FILEPATH = "ner_trained_model.h5"


CLASS_MAP = {
    0: 'PER',
    1: 'ORG',
    2: 'LOC',
    3: 'MISC',
    4: 'O'
}


def _get_dataset(file):
    """
    Read the dataset in `file` line-by-line and return a list of sentences.

    :param file: A String containing the path to the text document containing data.

    return: A list of sentences, where each sentence is a list of tuples:
            (<current word>, <part of speech tag>, <chunk tag>, <name entity tag>)
    """

    dataset = []

    with open(file, "r") as file:

        sentence = []

        # Read all lines one by one
        for line in file:
            line_items = line.split()

            # Ignore DOCSTART lines
            if 'DOCSTART' in line:
                continue

            # If empty row, add sentence to dataset and re-initialise it
            if len(line_items) == 0:

                if len(sentence) > 0:
                    dataset.append(sentence)
                sentence = []
                continue

            # If row is not empty add word to sentence
            if len(line_items) > 0:
                sentence.append(tuple(line_items))

    return dataset


def _preprocess_outputs(dataset):
    """
    Creates a targets matrix of shape (NUM_SENTENCES, MAX_SENTENCE_LENGTH, OUTPUT_DIMENSION)

    :param dataset: A list of tuples (<current word>, <part of speech tag>, <chunk tag>, <name entity tag>)
    containing the whole data in our dataset

    returns: A numpy 3D matrix for the model's `targets`
    """

    NUM_SENTENCES = len(dataset)

    # Initialise targets to zeros
    targets = np.zeros((NUM_SENTENCES, MAX_SENTENCE_LENGTH, OUTPUT_DIMENSION))

    # Loop over each sentence in the dataset, and each word in sentences
    for j, sentence in enumerate(dataset):
        for i, word in enumerate(sentence):

            # Only add the first MAX_SENTENCE_LENGTH words
            if i < MAX_SENTENCE_LENGTH:

                # Assign a target output for each word in the sentence
                label = word[3]
                if 'PER' in label:
                    targets[j, i, :] = [1, 0, 0, 0, 0]
                elif 'ORG' in label:
                    targets[j, i, :] = [0, 1, 0, 0, 0]
                elif 'LOC' in label:
                    targets[j, i, :] = [0, 0, 1, 0, 0]
                elif 'MISC' in label:
                    targets[j, i, :] = [0, 0, 0, 1, 0]
                elif label == 'O':
                    targets[j, i, :] = [0, 0, 0, 0, 1]
                else:
                    targets[j, i, :] = [0, 0, 0, 0, 0]

    return targets


def _preprocess_inputs(dataset_, word_index):
    """
    Loops through all words in all sentences and:
    (1) converts them to indices
    (2) adds padding to a length of MAX_SENTENCE_LENGTH

    :param dataset_: A list of tuples (<current word>, <part of speech tag>, <chunk tag>, <name entity tag>)
    containing the whole data in our dataset
    :param word_index: A dictionary with (<word>, <integer number>) (key, value) pairs

    return: A numpy matrix for the inputs with shape (NUM_SENTENCES, MAX_SENTENCE_LENGTH)
    """

    # initialise a list for the input data
    inputs = []

    # Loop through sentences
    for i, sentence in enumerate(dataset_):
        sentence = [word[0] for word in sentence]
        input_ = []

        # Loop through words in sentence
        for word in sentence:
            word = word.lower()
            word_num = word_index.get(word)
            if word_num is None:
                word_num = word_index.get('unk')
            input_.append(word_num)
        inputs.append(input_)

    # Pad sentences to some maximum length. Padding and truncation is done at the back of the sentence
    inputs = pad_sequences(inputs, maxlen=MAX_SENTENCE_LENGTH, padding='post', truncating='post')

    return inputs


def _load_trained_model(filepath='ner_trained_model.h5'):
    if not os.path.isfile(filepath):
        print("'ner_trained_model.h5' file is missing! Please put the file in the project's main folder.")
        return None
    return keras.models.load_model(filepath)


def _load_word_index(filepath='word_index.json'):
    if not os.path.isfile(filepath):
        print("'word_index.json' file is missing! Please put the file in the project's main folder.")
        return None

    with open(filepath, 'r') as file:
        word_index = json.load(file)
    return word_index


def _calculate_metrics_sentence(target_sentence, pred_sentence):

    # Filter out padding - this is where the target word is [0, 0, 0, 0, 0]
    for index, word in enumerate(target_sentence):
        if np.sum(word == np.array([0, 0, 0, 0, 0])) == OUTPUT_DIMENSION:
            break
    target_sentence = target_sentence[:index]
    pred_sentence = pred_sentence[:index]

    # Initialise results dictionary
    results = {}
    classes = list(range(target_sentence.shape[1]))
    for class_ in classes:
        results[class_] = {'tp': 0, 'fp': 0, 'fn': 0}

    # Get indices of target and predicted name entity class
    target_indices = np.argmax(target_sentence, axis=1)
    pred_indices = np.argmax(pred_sentence, axis=1)

    # Go through each target and prediction and sum up metrics
    for targ, pred in zip(target_indices,pred_indices):
        if targ == pred:
            results[targ]['tp'] += 1
            continue
        else:
            results[targ]['fn'] += 1
            results[pred]['fp'] += 1

    return results


def _calculate_metrics_total(model, test_data):
    test_x = test_data[0]
    test_y = test_data[1]

    preds = model.predict(test_x)

    # Initialise results dictionary
    results = {}
    classes = list(range(test_y[0].shape[1]))
    for class_ in classes:
        results[class_] = {'tp': 0, 'fp': 0, 'fn': 0}

    # Calculate metrics for each sentence one by one and add up results
    for target_sentence, pred_sentence in zip(test_y, preds):
        res_sentence = _calculate_metrics_sentence(target_sentence, pred_sentence)

        # Sum up results
        for class_ in classes:
            results[class_]['tp'] += res_sentence[class_]['tp']
            results[class_]['fp'] += res_sentence[class_]['fp']
            results[class_]['fn'] += res_sentence[class_]['fn']

    return results


def predict_on_test_set(filepath):

    # Calculate and print metrics
    results = _calculate_metrics_total(model, (inputs, targets))
    for label in results.keys():
        precision = results[label]['tp'] / (results[label]['tp'] + results[label]['fp'])
        recall = results[label]['tp'] / (results[label]['tp'] + results[label]['fn'])
        f1_score = 2 / (1 / precision + 1 / recall)
        support = 0
        for target in targets:
            support += np.sum(np.argmax(target) == label)
        print("%4s: precision = %5.3f recall = %5.3f f1_score = %5.3f support = %6d" %
              (
                  CLASS_MAP[label],
                  precision,
                  recall,
                  f1_score,
                  support
              ))


def _sentence_to_vector(sentence):

    # Import our word dictionary and tokenize the sentence
    word_index = _load_word_index(filepath='word_index2.json')
    tokens = word_tokenize(sentence)

    # Map word tokens into a vector of numbers
    tokens_num = [word_index.get(token.lower(), 0) for token in tokens]
    vector = np.zeros(shape=(1, MAX_SENTENCE_LENGTH))
    vector[0, : len(tokens_num)] = tokens_num

    return vector, tokens


def predict_on_texts(texts):

    for sentence in texts:
        vector, clean_sentence = _sentence_to_vector(sentence)
        preds = model.predict(vector)
        preds = preds[:, :len(clean_sentence)]
        pred_output_classes = np.argmax(preds[0], axis=1)

        for word, pred in zip(clean_sentence, pred_output_classes):
            print(word, end=" ")
            print("[%s]" % CLASS_MAP[pred], end=" ")
        print()


"""
------------------------------------------------------------------
The following code will run with importing the script
------------------------------------------------------------------
"""

# Load trained model
model = _load_trained_model(MODEL_FILEPATH)

# Create a dataset from text file
dataset = _get_dataset(DATASET_FILEPATH)

# Load dictionaries for mapping words to numbers and vice versa
word_index = _load_word_index(filepath=WORD_INDEX_FILEPATH)

# Process output labels
targets = _preprocess_outputs(dataset)

# Process input data
inputs = _preprocess_inputs(dataset, word_index)

"""
------------------------------------------------------------------
"""

if __name__ == '__main__':

    """
    Test the script here...
    """

    pass
    # texts = ["Hello how $34 are you doing, have you been to miami before? How about tokyo!?",
    #          "My name is Mihajlo, what's your name?",
    #          "I work at Imperial College London, which is located in London, UK",
    #          "Inter will beat Juventus 3:0 on Friday"]

    # predict_on_texts(texts)
    # predict_on_test_set(filepath=DATASET_FILEPATH)

