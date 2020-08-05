import argparse
import json
import numpy as np
import os
import string

def transform_to_next_character_prediction(args):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(current_dir, args.datadir)

    for subdir, dirs, files in os.walk(data_dir):
        for filename in files:
            filepath = subdir + os.sep + filename
            transformed_filepath = subdir + os.sep + "next-character-" + filename

            if filepath.endswith(".json"):
                print("Processing %s now." % filepath)
                users, user_data = load_data(filepath)

                num_samples = []
                processed_users = set()

                for user in users:
                    # At least one user (4482) can be found twice in users
                    if (user not in processed_users):
                        data_x, data_y = to_next_character_prediction(user_data[user])
                        user_data[user]['x'] = data_x
                        user_data[user]['y'] = data_y
                        processed_users.add(user)
                    num_samples.append(user_data[user]['y'])
                
                save_data(users, num_samples, user_data, transformed_filepath)


def to_next_character_prediction(user_data, seq_length=80):
    """Converts reddit natural language processing format into next character prediction as in shakespeare.

    Args:
        user_data: user_data of one specific user
        seq_length: length of strings in X
    """

    # Also remove ", because shakespeare data does not use "
    TOKENS_TO_REMOVE = ['<BOS>', '<EOS>', '<PAD>', '"']
    
    raw_text = []
    data_y = user_data['y']

    # Reconstruct the possible original text
    raw_data = [np.array(x['target_tokens'], dtype=str).flatten() for x in data_y]
    for x in raw_data:
        x = [x for x in x if x not in TOKENS_TO_REMOVE]
        raw_text.extend(x)
    raw_text = ''.join([' ' + x if x not in string.punctuation else x for x in raw_text])
    raw_text = raw_text.strip()

    # Preprocess the "original" text to fit next character format
    dataX = []
    dataY = []
    for i in range(0, len(raw_text) - seq_length, 1):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        dataX.append(seq_in)
        dataY.append(seq_out)
    return dataX, dataY

def load_data(filepath):
    with open(filepath) as inf:
        data = json.load(inf)

    users = data['users']
    user_data = data['user_data']

    return users, user_data

def save_data(users, num_samples, user_data, filepath):
    all_data = {}
    all_data['users'] = users
    all_data['num_samples'] = num_samples
    all_data['user_data'] = user_data

    with open(filepath, 'w') as outfile:
        json.dump(all_data, outfile)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--datadir',
        help='Specify where to search for reddit data. Default: "data"',
        default='data',
        type=str,
        required=False)
    
    return parser.parse_args()

args = parse_args()
transform_to_next_character_prediction(args)
