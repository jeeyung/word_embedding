import re

def preprocess_word(word):
    word = word.lower()
    word = "".join(re.findall("[a-z]+", word))
    return word