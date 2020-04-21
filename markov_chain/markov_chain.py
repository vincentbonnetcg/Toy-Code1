"""
@author: Vincent Bonnet
@description : Markov Chain
"""

import re
import pandas as pd
import numpy as np
import random
import os

ORDER = 2

def prepare_string(txt):
    # not perfect but good enough
    return re.sub('[^A-Za-z0-9]+', ' ', txt)

def load_names_from_poetry_foundations():
    # from https://www.kaggle.com/tgdivy/poetry-foundation-poems  should be downloaded
    filename = os.path.join(os.getcwd(), "kaggle_poem_dataset.csv")
    data = pd.read_csv(filename)
    first_names = set()
    family_names = set()
    num_rows = len(data['Author'])
    for txt in data['Author']:
        split_text = prepare_string(txt).split()
        if len(split_text) == 2:
            first_names.add(split_text[0])
            family_names.add(split_text[1])

    return list(first_names), list(family_names)

def create_transition_matrix(names):
    # create transition matrix from first names
    transition = {}
    possible_values = set()
    for name in names:
        for i in range(0, len(name) - ORDER):
            key = tuple(name[i:i+ORDER])
            values = transition.get(key, [])
            values.append(name[i+ORDER])
            transition[key] = values
            possible_values.add(name[i+ORDER])

    # compute the probabilities and transition matrix
    transition_matrix = {}
    possible_values = list(possible_values)
    for key, values in transition.items():
        propability_row = np.zeros(len(possible_values))
        for value in values:
            index = possible_values.index(value)
            propability_row[index] += 1.0

        propability_row /= len(values)
        transition_matrix[key] = propability_row

    return transition_matrix, possible_values


def generate_txt(start_key, transition_matrix, possible_values, num_words):
    txt = []
    for i in range(len(start_key)):
        txt.append(start_key[i])

    for i in range(num_words):
        key = tuple(txt[-ORDER:])
        probabilities = transition_matrix.get(key, None)
        if not probabilities is None:
            value = np.random.choice(possible_values,replace=True,p=probabilities)
            txt.append(value)
        else:
            break

    print(''.join(txt))

def main():
    random.seed()
    first_names, family_names = load_names_from_poetry_foundations()
    # generate first name
    start_key = random.choice(first_names)[:ORDER]
    transition_matrix, possible_values = create_transition_matrix(first_names)
    generate_txt(start_key, transition_matrix, possible_values, 5)
    # generate family name
    start_key = random.choice(first_names)[:ORDER]
    transition_matrix, possible_values = create_transition_matrix(family_names)
    generate_txt(start_key, transition_matrix, possible_values, 5)

if __name__ == '__main__':
    main()
