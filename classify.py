#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import collections
import math
import pickle

import util


class NaiveBayesModel(object):

    def __init__(self, class_freqs, word_freqs, class_sizes, vocabulary):
        self.class_freqs = class_freqs
        self.word_freqs = word_freqs
        self.class_sizes = class_sizes
        self.vocabulary = vocabulary
        self.vocab_size = len(vocabulary)

    def doc_in_class_probabilities(self, tokens, class_labels):
        """
        :return: a list of (class label, log probability) tuples
        """

        probabilities = []

        for class_label in class_labels:
            p = math.log(self.class_freqs[class_label])

            for word in tokens:
                word_p = self.word_freqs[(class_label, word)] + 1

                word_p = word_p / float(self.vocab_size + self.class_sizes[class_label])

                p += math.log(word_p)

            probabilities.append((class_label, p))

        return probabilities

    def classify(self, input_path, encoding):
        """
        :return: a tuple of (class label, weights) where ``weights`` is the
            list of (class label, weight) tuples
        """
        
        with open(input_path, "r") as input_file:
            doc_tokens = util.tokenize(util.get_text(input_file, encoding))

        weights = self.doc_in_class_probabilities(
            doc_tokens,
            self.class_freqs.keys()
        )

        best_weight = None

        for label, weight in weights:
            if best_weight is None or weight > best_weight:
                best_weight = weight
                best_label = label

        return (best_label, weights)


def train(input_paths, encoding, output_path):
    """
    :param input_paths: a list of one string representing the input path to a
        metadata file. Each line of the file contains class name and path to
        input file separated by whitespace.
    """
    input_metadata = input_paths[0]

    doc_tokens = []
    doc_labels = []

    class_freqs = collections.defaultdict(int)
    word_freqs = collections.defaultdict(int)
    class_sizes = collections.defaultdict(int)

    vocabulary = set()

    with open(input_metadata, "r") as metadata_file:
        for line in metadata_file:
            doc_label, doc_path = line.strip().split()

            with open(doc_path, "r") as doc_file:
                tokens = util.tokenize(util.get_text(doc_file, encoding))

            doc_labels.append(doc_label)
            doc_tokens.append(tokens)

            class_freqs[doc_label] += 1
            class_sizes[doc_label] += len(doc_tokens)

            for word in tokens:
                vocabulary.add(word)
                word_freqs[(doc_label, word)] += 1

    class_freqs = {c : f / float(len(doc_tokens)) for c, f in class_freqs.iteritems()}

    model = NaiveBayesModel(class_freqs, word_freqs, class_sizes, vocabulary)

    with open(output_path, "w") as output_file:
        pickle.dump(model, output_file)


def classify(input_paths, encoding, output_path, model_path):
    with open(model_path, "r") as model_file:
        model = pickle.load(model_file)

    for input_path in input_paths:
        best_label, weights = model.classify(input_path, encoding)

        print "{path}: {label} ({weights})".format(
            path=input_path,
            label=best_label,
            weights=weights
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Naive Bayes on documents or classify using an existing model"
    )

    parser.add_argument("-t", "--train",
        help=u"If present, will train on documents",
        action="store_true"
    )

    parser.add_argument("-c", "--classify",
        help=u"If present, will classify a new document",
        action="store_true"
    )

    parser.add_argument("-i", "--input-paths",
        help=u"Path to training metadata file or documents to classify",
        metavar="PATH",
        type=str,
        nargs='+',
        required=True
    )

    parser.add_argument("-m", "--model-path",
        help=u"Path to model file",
        metavar="PATH",
        type=str,
    )

    parser.add_argument("-o", "--output-path",
        help=u"Output path",
        metavar="PATH",
        type=str
    )

    parser.add_argument("-e", "--encoding",
        help=u"Encoding to use for text files (UTF-8)",
        default="utf-8",
        type=str
    )

    args = parser.parse_args()

    if args.train:
        train(args.input_paths, args.encoding, args.output_path)
    elif args.classify:
        classify(
            args.input_paths,
            args.encoding,
            args.output_path,
            args.model_path
        )
