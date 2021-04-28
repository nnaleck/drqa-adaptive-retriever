#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Interactive mode for the tfidf DrQA retriever module."""

import argparse
import code
import prettytable
import logging
from drqa import retriever
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--csv', type=str, default=None)
args = parser.parse_args()

logger.info('Initializing ranker...')
ranker = retriever.get_class('tfidf')(tfidf_path=args.model)

# Loading questions
if args.csv:
    df = pd.read_csv(args.csv)
    df = df[df["Full_Text"].notna()]
    df = df.reset_index()


# ------------------------------------------------------------------------------
# Drop in to interactive
# ------------------------------------------------------------------------------


def process(query, k=1):
    doc_names, doc_scores = ranker.closest_docs(query, k)
    table = prettytable.PrettyTable(
        ['Rank', 'Doc Id', 'Doc Score']
    )
    for i in range(len(doc_names)):
        table.add_row([i + 1, doc_names[i], '%.5g' % doc_scores[i]])
    print(table)


def calculate_top_n_ratio(top_n=3):
    nb_included_in_top_n = 0
    positions_sum = 0

    for i in df.index:
        question_query = df["Question"][i]
        pmid_query = str(df["Document_ID"][i])

        doc_names, doc_scores = ranker.closest_docs(question_query, top_n)

        for j in range(len(doc_names)):
            pmid_result = doc_names[j]
            if pmid_result == pmid_query:
                nb_included_in_top_n += 1
                positions_sum += (j + 1)

    return nb_included_in_top_n, nb_included_in_top_n / df.shape[0], positions_sum / nb_included_in_top_n


banner = """
Interactive TF-IDF DrQA Retriever
>> process(question, k=1)
>> usage()
"""


def usage():
    print(banner)


code.interact(banner=banner, local=locals())
