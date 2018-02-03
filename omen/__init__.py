import random
import nltk
from collections import defaultdict
from nltk.tokenize.moses import MosesDetokenizer

detokenizer = MosesDetokenizer()

class Omen():
    def __init__(self, state_size=3):
        self.START_TOKEN = "@@_START_TOKEN_@@"
        self.END_TOKEN = "@@_END_TOKEN_@@"
        self.STATE_SIZE = state_size

        self.states = defaultdict(lambda: defaultdict(int))
        self.beginnings = defaultdict(int)

    @staticmethod
    def weighted_choice(words):
        number = random.random() * sum(words.values())
        for key, weight in words.items():
            number -= weight
            if number < 0:
                return key

    def train(self, corpus):
        sentences = nltk.sent_tokenize(corpus)
        sentences = [[self.START_TOKEN, *nltk.word_tokenize(sentence), self.END_TOKEN] for sentence in sentences]

        for sentence in sentences:
            self.beginnings[tuple(sentence[:self.STATE_SIZE])] += 1
            for index, word in enumerate(sentence):
                state = tuple(sentence[index - self.STATE_SIZE:index])
                self.states[state][word] += 1

    def generate(self):
        tokens = list(Omen.weighted_choice(self.beginnings))
        next_word = tokens[-1]
        while next_word != self.END_TOKEN:
            state = tuple(tokens[-self.STATE_SIZE:])
            next_word = Omen.weighted_choice(self.states[state])
            tokens.append(next_word)
        
        return detokenizer.detokenize(tokens[1:-1], return_str=True)
