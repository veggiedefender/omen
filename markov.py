import random
import nltk
from collections import defaultdict
from nltk.tokenize.moses import MosesDetokenizer

START = "@@_START_@@"
END = "@@_END_@@"
STATE_SIZE = 3

def weighted_choice(words):
    number = random.random() * sum(words.values())
    for key, weight in words.items():
        number -= weight
        if number < 0:
            return key

with open("corpus.txt") as f:
    corpus = f.read()

sentences = nltk.sent_tokenize(corpus)
sentences = [[START, *nltk.word_tokenize(sentence), END] for sentence in sentences]

states = defaultdict(lambda: defaultdict(int))
beginnings = []
for sentence in sentences:
    beginnings.append(tuple(sentence[:STATE_SIZE]))
    for index, word in enumerate(sentence):
        key = tuple(sentence[index - STATE_SIZE:index])
        states[key][word] += 1

tokens = list(random.choice(beginnings))
curr = tokens[-1]
while curr != END:
    key = tuple(tokens[-STATE_SIZE:])
    curr = weighted_choice(states[key])
    tokens.append(curr)

detokenizer = MosesDetokenizer()
print(detokenizer.detokenize(tokens[1:-1], return_str=True))
