"""A library that includes methods for training and generating content
using markov chains
"""
import random
from collections import defaultdict
import dill as pickle
import nltk
from nltk.tokenize.moses import MosesDetokenizer

__DETOKENIZER = MosesDetokenizer()

def sentence_tokenize(corpus):
    """Tokenizes a corpus into sentences

    Uses nltk.sent_tokenize() to split a body of text into sentences

    Args:
        corpus: a string containing the corpus to parse

    Returns:
        a list of strings containing the corpus's sentences
    """
    return nltk.sent_tokenize(corpus)

def word_tokenize(sentence):
    """Tokenizes a sentence into words

    Uses nltk.word_tokenize() to split a sentence into words

    Args:
        sentence: a string containing a sentence

    Returns:
        a list of strings containing the sentence's words
    """
    return nltk.word_tokenize(sentence)

def detokenize(tokens):
    """Combines a list of tokens back into a sentence

    Uses nltk.MosesTokenizer to join strings back together with
    proper spacing

    Args:
        tokens: a list of strings containing tokens (words, punctuation)

    Returns:
        a string that is the result of combining the tokens
    """
    return __DETOKENIZER.detokenize(tokens, return_str=True)

def weighted_choice(words):
    """Randomly chooses a word from a dict of words and frequencies

    Chooses a random number between 0 and the total frequency of all
    words given, and finds which bucket the number falls in

    Args:
        words: a dict of words paired with frequencies. For example:
        "one fish two fish red fish blue fish"
        will look like:
        {
            "one": 1,
            "fish": 4,
            "two": 1,
            "red": 1,
            "blue": 1
        }

    Returns:
        the randomly chosen string
    """
    number = random.random() * sum(words.values())
    for key, weight in words.items():
        number -= weight
        if number < 0:
            return key

class Omen():
    """Class containing markov chain state and metadata

    Attributes:
        start_token: string marking the beginning of a sentence
        end_token: string marking the end of a sentence
        state_size: an integer number of words to consider when choosing
            the next state
        states: dict mapping states to possible next states
        beginnings: cache of starting tokens to initialize the chain
    """
    def __init__(self, state_size=3):
        self.start_token = "@@_START_TOKEN_@@"
        self.end_token = "@@_END_TOKEN_@@"
        self.state_size = state_size

        self.states = defaultdict(lambda: defaultdict(int))
        self.beginnings = defaultdict(int)

    def dump_states(self, file):
        """Dump state to a file

        Uses dill to save all states and metadata to an open file. Use
        it like this:
        >>> omen.dump_states(open("states.pkl", "wb"))

        Args:
            file: open file to dump data to
        """
        pickle.dump({
            "start_token": self.start_token,
            "end_token": self.end_token,
            "state_size": self.state_size,
            "states": self.states,
            "beginnings": self.beginnings
        }, file)

    def load_states(self, file):
        """Loads state from a file back into an omen object

        Uses dill to load a dump of states and metadata created by
        dump_states() from an open file. Use it like this:
        >>> omen.load_states(open("states.pkl", "rb"))

        Args:
            file: open file to load data from
        """
        states = pickle.load(file)
        self.start_token = states["start_token"]
        self.end_token = states["end_token"]
        self.state_size = states["state_size"]
        self.states = states["states"]
        self.beginnings = states["beginnings"]

    def train(self, corpus):
        """Read a corpus into a markov chain probability model

        Tokenizes the corpus and pairs states (tuples containing a
        state_size number of tokens) with the frequencies of the states
        that come next.

        Args:
            corpus: a string containing the corpus to represent in the
            markov chain
        """
        sentences = sentence_tokenize(corpus)
        sentences = [[self.start_token, *word_tokenize(sentence), self.end_token] for sentence in sentences]

        for sentence in sentences:
            self.beginnings[tuple(sentence[:self.state_size])] += 1
            for index, word in enumerate(sentence):
                state = tuple(sentence[index - self.state_size:index])
                self.states[state][word] += 1

    def generate(self):
        """Randomly generate a sentence using the markov chain

        Randomly pick some starting tokens, then iteratively pick random
        states that come next until we reach the end of a sentence

        Returns:
            a string containing a sentence generated using the markov
            chain
        """
        tokens = list(weighted_choice(self.beginnings))
        next_word = tokens[-1]
        while next_word != self.end_token:
            state = tuple(tokens[-self.state_size:])
            next_word = weighted_choice(self.states[state])
            tokens.append(next_word)

        return detokenize(tokens[1:-1])
