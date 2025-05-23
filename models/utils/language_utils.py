"""Utils for language models."""

import re
import numpy as np
import json


# ------------------------
# utils for shakespeare dataset

ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
NUM_LETTERS = len(ALL_LETTERS)


def _one_hot(index, size):
    '''returns one-hot vector with given size and value 1 at given index
    '''
    vec = [0 for _ in range(size)]
    vec[int(index)] = 1
    return vec


def letter_to_vec(letter):
    '''returns one-hot representation of given letter
    '''
    index = ALL_LETTERS.find(letter)
    return _one_hot(index, NUM_LETTERS)


def word_to_indices(word):
    '''returns a list of character indices

    Args:
        word: string
    
    Returns:
        indices: int list with length len(word)
    '''
    indices = []
    for c in word:
        indices.append(ALL_LETTERS.find(c))
    return indices


# ------------------------
# utils for combined shakespeare and reddit dataset (next-character)

# The first 80 letters may appear both in Reddit and Shakespeare dataset
# The last letter represents all additional characters/letters (which may appear in Reddit only)
ALL_LETTERS_NC = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}*"
NUM_LETTERS_NC = len(ALL_LETTERS_NC)

def letter_to_vec_nc(letter):
    '''returns one-hot representation of given letter
       All unknown characters will be mapped to the last item
    '''
    index = ALL_LETTERS_NC.find(letter)
    if (index == -1):
        index = NUM_LETTERS_NC - 1
    return _one_hot(index, NUM_LETTERS_NC)


def word_to_indices_nc(word):
    '''returns a list of character indices
    All unknown characters will be mapped to the last item

    Args:
        word: string
    
    Returns:
        indices: int list with length len(word)
    '''
    indices = []
    for c in word:
        index = ALL_LETTERS_NC.find(c)
        if (index == -1):
            index = NUM_LETTERS_NC - 1
        indices.append(index)
    return indices

# ------------------------
# utils for combined shakespeare and goethe dataset (poets)

# The last letter represents all additional characters/letters
ALL_LETTERS_POETS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜẞ[]abcdefghijklmnopqrstuvwxyzäöüß}*"
NUM_LETTERS_POETS = len(ALL_LETTERS_POETS)

def letter_to_vec_poets(letter):
    '''returns one-hot representation of given letter
       All unknown characters will be mapped to the last item
    '''
    index = ALL_LETTERS_POETS.find(letter)
    if (index == -1):
        index = NUM_LETTERS_POETS - 1
    return np.squeeze(np.array(_one_hot(index, NUM_LETTERS_POETS)))

def letter_to_index_poets(letter):
    '''returns one-hot representation of given letter
       All unknown characters will be mapped to the last item
    '''
    index = ALL_LETTERS_POETS.find(letter)
    if (index == -1):
        index = NUM_LETTERS_POETS - 1
    return index


def word_to_indices_poets(word):
    '''returns a list of character indices
    All unknown characters will be mapped to the last item

    Args:
        word: string
    
    Returns:
        indices: int list with length len(word)
    '''
    indices = []
    for c in word:
        index = ALL_LETTERS_POETS.find(c)
        if (index == -1):
            index = NUM_LETTERS_POETS - 1
        indices.append(index)
    return np.array(indices)

# ------------------------
# utils for sent140 dataset


def split_line(line):
    '''split given line/phrase into list of words

    Args:
        line: string representing phrase to be split
    
    Returns:
        list of strings, with each string representing a word
    '''
    return re.findall(r"[\w']+|[.,!?;]", line)


def _word_to_index(word, indd):
    '''returns index of given word based on given lookup dictionary

    returns the length of the lookup dictionary if word not found

    Args:
        word: string
        indd: dictionary with string words as keys and int indices as values
    '''
    if word in indd:
        return indd[word]
    else:
        return len(indd)


def line_to_indices(line, word2id, max_words=25):
    '''converts given phrase into list of word indices
    
    if the phrase has more than max_words words, returns a list containing
    indices of the first max_words words
    if the phrase has less than max_words words, repeatedly appends integer 
    representing unknown index to returned list until the list's length is 
    max_words

    Args:
        line: string representing phrase/sequence of words
        word2id: dictionary with string words as keys and int indices as values
        max_words: maximum number of word indices in returned list

    Returns:
        indl: list of word indices, one index for each word in phrase
    '''
    unk_id = len(word2id)
    line_list = split_line(line) # split phrase in words
    indl = [word2id[w] if w in word2id else unk_id for w in line_list[:max_words]]
    indl += [unk_id]*(max_words-len(indl))
    return indl


def bag_of_words(line, vocab):
    '''returns bag of words representation of given phrase using given vocab

    Args:
        line: string representing phrase to be parsed
        vocab: dictionary with words as keys and indices as values

    Returns:
        integer list
    '''
    bag = [0]*len(vocab)
    words = split_line(line)
    for w in words:
        if w in vocab:
            bag[vocab[w]] += 1
    return bag


def get_word_emb_arr(path):
    with open(path, 'r') as inf:
        embs = json.load(inf)
    vocab = embs['vocab']
    word_emb_arr = np.array(embs['emba'])
    indd = {}
    for i in range(len(vocab)):
        indd[vocab[i]] = i
    vocab = {w: i for i, w in enumerate(embs['vocab'])}
    return word_emb_arr, indd, vocab


def val_to_vec(size, val):
    """Converts target into one-hot.

    Args:
        size: Size of vector.
        val: Integer in range [0, size].
    Returns:
         vec: one-hot vector with a 1 in the val element.
    """
    assert 0 <= val < size
    vec = [0 for _ in range(size)]
    vec[int(val)] = 1
    return vec
