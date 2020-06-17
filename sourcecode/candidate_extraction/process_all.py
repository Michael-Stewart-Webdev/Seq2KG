#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 12:48:55 2019

@author: Majigsuren Enkhsaikhan
"""

import os
import pandas as pd
import csv

from triples_from_text import extract_triples

def process_all():
    abspath = os.path.abspath('')
    os.chdir(abspath)
    
    df = pd.read_csv(r'../../datasets/icdm_contest_data.csv', encoding='utf-8', header=0, quoting=csv.QUOTE_ALL) #, skipinitialspace=True)

    all_docs_triples = []
    for index, row in df.iterrows():        
        mytriples = extract_triples(row[1])
        for s, p, o in mytriples:
            all_docs_triples.append([row[2], row[0], s, p, o])
        print("\rStart the process for %d articles... %d/%d complete" % (len(df), index + 1, len(df)), end="")
    print()

    with open('../../submission.csv', mode='w', newline='', encoding='utf-8') as uwa_file:
        csv_writer = csv.writer(uwa_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['industry', 'index', 's1', 'r', 's2'])
        csv_writer.writerows(all_docs_triples)


# Reads data file and creates the submission.csv file
if __name__ == "__main__":
    process_all()
    print("Finished the process. 'submission.csv' file is created.")
