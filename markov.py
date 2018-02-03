import nltk
from nltk.tokenize.moses import MosesDetokenizer
import random

START = "@@_START_@@"
END = "@@_END_@@"
STATE_SIZE = 3

with open("corpus.txt") as f:
    corpus = f.read()

sentences = nltk.sent_tokenize(corpus)
sentences = [[START, *nltk.word_tokenize(sentence), END] for sentence in sentences]

states = {}
beginnings = []
for sentence in sentences:
    beginnings.append(tuple(sentence[:STATE_SIZE]))
    for index, word in enumerate(sentence):
        key = tuple(sentence[index - STATE_SIZE:index])
        try:
            states[key].append(word)
        except KeyError:
            states[key] = [word]

tokens = list(random.choice(beginnings))
curr = tokens[-1]
while curr != END:
    key = tuple(tokens[-STATE_SIZE:])
    curr = random.choice(states[key])
    tokens.append(curr)

detokenizer = MosesDetokenizer()
print(detokenizer.detokenize(tokens[1:-1], return_str=True))
