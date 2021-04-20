"""
This script converts our dataset in order to build an SQLite
database using DrQA's build_db.py script.
"""

import pandas as pd
import json

df = pd.read_csv('fulltext_dataset_pdf.csv')
df = df[df["Full_Text"].notna()]
df = df.drop(columns=["Full_Text", "Document_ID", "Contexts", "Long_Answer"])

with open('pubmed-dataset.txt', 'w') as file:
    for index in df.index:
        question = df["Question"][index]
        answer = [df["final_decision"][index]]
        file.write(json.dumps({'question': question, 'answer': answer}))
        file.write('\n')
