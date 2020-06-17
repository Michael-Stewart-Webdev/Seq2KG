# Redcoat Parser

Parses Redcoat annotations into datasets for the Candidate Filtering and End-to-end model.

Usage:

```
   $ python create_datasets.py
```

First ensure that the `annotations.json` file is present under `data/cateringServices` so that it may be processed.

In order to use this script, the BERT server must be running. This is because the triples are evaluated against the ground truths using the cosine distance between the averaged word embedding vectors of the head, rel, and tail. There are instructions under `candidate_filtering` for how to set up and run the BERT server.