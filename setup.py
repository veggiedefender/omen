from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='omen',
      version='1.4',
      description='Simple markov chain library',
      long_description=readme(),
      keywords='markov chain nlp',
      url='https://github.com/veggiedefender/omen',
      author='Jesse Li',
      author_email='jessejesse123@gmail.com',
      license='MIT',
      packages=['omen'],
      install_requires=['nltk', 'dill'],
      setup_requires=['nltk'],
      zip_safe=True)

import nltk
nltk.download('punkt')
nltk.download('perluniprops')
