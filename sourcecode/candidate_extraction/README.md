# Candidate Extraction 

We found that the best graph edit distance scores were obtained using our Candidate Extraction model.
Candidate Extraction is a model that takes text as an input and returns triples as an output. This Python model applies existing well established NLP frameworks such as SpaCy, NLTK, NeuralCoref and NetworkX for its purposes.

The main stages:

1. Extracting all triples from a given text, using SpaCy. 
    * Extract entities (proper nouns + noun phrase chunks)
        * Proper nouns (BYD, Jamie Oliver, Beijing, Shanghai International Automobile Industry Exhibition)
        * Noun phrase chunks (Dynasty series vehicles, bankrupcy protection)
    * Extract relations based on verbs, conjunctions and adpositions between entities
        * Verbs convey actions (open, showcase, close), occurrences (happen, become), or a state of being (be, exist).
        * Conjunction words connect clauses or sentences, also to coordinate words in the same clause (and, but, with).
        * Adpositions are prepositions and postpositions that occur before or after entities (in, to, at, during).
    * Solve coreferences using NeuralCoref, which is a pipeline extension for spaCy 2.1+. NeuralCoref annotates and resolves coreference clusters. 

2. Create a graph (using NetworkX) from the triples and then:
    * using the shortest paths, create edges between proper nouns if the relation is adpositions;
    * set weights on the nodes according to their degree

Given input sentences:

    "BYD debuted its E-SEED GT concept car and Song Pro SUV alongside its all-new e-series models at the Shanghai International Automobile Industry Exhibition.
	The company also showcased its latest Dynasty series of vehicles, which were recently unveiled at the company's spring product launch in Beijing."

Output triples:
   
    ["BYD", "at", "Shanghai International Automobile Industry Exhibition"]
    ['BYD', 'debuted', 'E-SEED GT concept car']
    ["BYD", "debuted", "Song Pro SUV"]
    ["BYD", "debuted, "all-new e-series models"]
    ["BYD", "showcased, "Dynasty series vehicles"]
    ["company's spring product launch", "in", "Beijing"]
    ["BYD", "with", "Beijing"]
    ['E-SEED GT concept car', 'at', 'Shanghai International Automobile Industry Exhibition']
    ['Song Pro SUV', 'at', 'Shanghai International Automobile Industry Exhibition']
    ['all-new e-series models', 'at', 'Shanghai International Automobile Industry Exhibition']

##### Model drawbacks/future work
Candidate Extraction model needs improvement on coreference resolution and filtering out the less significant nodes to keep the few important triples.

##### Setup instructions
The model is implemented in Python 3.

The following Python packages are required:
    * python==3,
    * networkx==2.1
    * nltk==3.2.5
    * pygraphviz==1.3
    * spacy==2.1.0
    * matplotlib==2.1.2
    * pandas==0.22.0
    * neuralcoref==4.0
    * nltk.download('stopwords')

The easiest way to install all required libraries at once is to run in Candidate Extraction folder:

   ```
   $ pip install -r requirements.txt
   ```

##### Model Usage
Candidate Extraction model reads the ICDM input text file and creates a list of triples and saves them in `submission.csv` file.

   ```
   $ python3 process_all.py
   ```