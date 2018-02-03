import random
from collections import defaultdict
import dill as pickle
import nltk
from nltk.tokenize.moses import MosesDetokenizer


class Omen():
    def __init__(self, state_size=3):
        self.detokenizer = MosesDetokenizer()

        self.start_token = "@@_START_TOKEN_@@"
        self.end_token = "@@_END_TOKEN_@@"
        self.state_size = state_size

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
            "start_token": self.start_token,
            "end_token": self.end_token,
            "state_size": self.state_size,
            "states": self.states,
            "beginnings": self.beginnings
        }, file)

    def load_states(self, file):
        states = pickle.load(file)
        self.start_token = states["start_token"]
        self.end_token = states["end_token"]
        self.state_size = states["state_size"]
        self.states = states["states"]
        self.beginnings = states["beginnings"]

    def train(self, corpus):
        sentences = nltk.sent_tokenize(corpus)
        sentences = [[self.start_token, *nltk.word_tokenize(sentence), self.end_token] for sentence in sentences]

        for sentence in sentences:
            self.beginnings[tuple(sentence[:self.state_size])] += 1
            for index, word in enumerate(sentence):
                state = tuple(sentence[index - self.state_size:index])
                self.states[state][word] += 1

    def generate(self):
        tokens = list(Omen.weighted_choice(self.beginnings))
        next_word = tokens[-1]
        while next_word != self.end_token:
            state = tuple(tokens[-self.state_size:])
            next_word = Omen.weighted_choice(self.states[state])
            tokens.append(next_word)

        return self.detokenizer.detokenize(tokens[1:-1], return_str=True)
