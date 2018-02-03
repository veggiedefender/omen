import random
import nltk
from collections import defaultdict
import dill as pickle
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

    def dump_states(self, file):
        pickle.dump({
            "START_TOKEN": self.START_TOKEN,
            "END_TOKEN": self.END_TOKEN,
            "STATE_SIZE": self.STATE_SIZE,
            "states": self.states,
            "beginnings": self.beginnings
        }, file)

    def load_states(self, file):
        states = pickle.load(file)
        self.START_TOKEN = states["START_TOKEN"]
        self.END_TOKEN = states["END_TOKEN"]
        self.STATE_SIZE = states["STATE_SIZE"]
        self.states = states["states"]
        self.beginnings = states["beginnings"]

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
