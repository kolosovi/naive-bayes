#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import string

import nltk


def get_text(input_file, encoding):
    return input_file.read().decode(encoding).lower()


def tokenize(text):
    russian_stopwords = set(nltk.corpus.stopwords.words('russian'))

    punctuation = re.compile(
        r"[" + string.punctuation + string.ascii_letters + string.digits + "]"
    )

    words = nltk.word_tokenize(text)

    words = [word for word in words if word not in russian_stopwords]

    return [word for
            word in words
            if not re.search(punctuation, word)]
