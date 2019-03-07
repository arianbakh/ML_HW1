import numpy as np
import os
import pickle
import sys


# Directory Settings
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
TRAINING_DATA_DIR = os.path.join(BASE_DIR, 'training_data')
BINARY_DIR = os.path.join(TRAINING_DATA_DIR, 'binary')
BIPOLAR_DIR = os.path.join(TRAINING_DATA_DIR, 'bipolar')
W_TRANSPOSE_PATH = os.path.join(BASE_DIR, 'w_transpose')
TEST_PATH = os.path.join(BASE_DIR, 'test_input.txt')


# Parameters
INITIAL_VALUE_MAX = 0.1
MAX_EPOCHS = 50
BINARY_THETA = 0.5
BIPOLAR_THETA = 0
ALPHA = 1


def _save_w_transpose(w_transpose):
    pass  # TODO


def _load_w_transpose():
    pass  # TODO


def _get_input_vector(input_file_path):
    input_list = []

    with open(input_file_path, 'r') as input_file:
        for line in input_file.readlines():
            line = line.strip()
            if line:
                input_strings = line.split()
                for input_string in input_strings:
                    input_list.append(int(input_string))

    input_vector = np.zeros((len(input_list) + 1, 1))
    input_vector[0, 0] = 1  # bias
    for j, input_number in enumerate(input_list):
        input_vector[j + 1, 0] = input_number

    return input_vector


def _get_training_pairs(mode):
    if mode == 'binary':
        training_data_dir = BINARY_DIR
    else:
        training_data_dir = BIPOLAR_DIR

    label_names = os.listdir(training_data_dir)
    for i, label_name in enumerate(label_names):
        label_vector = np.full((len(label_names), 1), -1)
        label_vector[i, 0] = 1

        label_dir = os.path.join(training_data_dir, label_name)
        for input_file_name in os.listdir(label_dir):
            input_file_path = os.path.join(label_dir, input_file_name)
            input_vector = _get_input_vector(input_file_path)

            yield (input_vector, label_vector)


_bipolar_activation = np.vectorize(lambda x: -1 if x < -1 * BIPOLAR_THETA else (0 if -1 * BIPOLAR_THETA <= x <= BIPOLAR_THETA else 1))


def _get_initial_w_transpose(training_pairs):
    width = len(training_pairs[0][0])
    height = len(training_pairs[0][1])
    return np.random.rand(height, width) * INITIAL_VALUE_MAX


def _execute_epoch(w_transpose, training_pairs):
    number_of_mismatches = 0
    for input_vector, label_vector in training_pairs:
        y_in = np.matmul(w_transpose, input_vector)
        y_out = _bipolar_activation(y_in)
        difference_vector = np.zeros(y_out.shape)
        for i in range(len(y_out)):
            if label_vector[i, 0] != y_out[i, 0]:
                number_of_mismatches += 1
                difference_vector[i, 0] = label_vector[i, 0]
        delta_w = ALPHA * np.matmul(difference_vector, input_vector.T)
        w_transpose += delta_w

    return w_transpose, number_of_mismatches


def train(mode):
    training_pairs = list(_get_training_pairs(mode))

    w_transpose = _get_initial_w_transpose(training_pairs)
    number_of_mismatches = -1

    epoch = 0
    while number_of_mismatches != 0:
        w_transpose, number_of_mismatches = _execute_epoch(w_transpose, training_pairs)

        epoch += 1

        print('Epoch', epoch)
        print(number_of_mismatches)

        if epoch > MAX_EPOCHS:
            break

    return w_transpose


def test(w_transpose):
    input_vector = _get_input_vector(TEST_PATH)
    y_in = np.matmul(w_transpose, input_vector)
    y_out = _bipolar_activation(y_in)
    print(y_out)


def run(mode):
    w_transpose = train(mode)
    test(w_transpose)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Invalid Number of Arguments')
        exit()

    if sys.argv[1] == 'binary':
        run('binary')
    elif sys.argv[1] == 'bipolar':
        run('bipolar')
    else:
        print('Invalid Argument')
        exit()
