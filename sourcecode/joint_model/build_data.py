from torch.utils.data import DataLoader
import sys
import numpy as np
import pickle as pkl
import os

import codecs, jsonlines, json, random

from logger import logger

import data_utils as dutils
from data_utils import Vocab, CategoryHierarchy, EntityTypingDataset, tokens_to_wordpieces

from config import Config
cf = Config()

import spacy
from spacy.tokens import Doc
import neuralcoref

MAX_SENT_LEN = -1


# A class for holding one sentence from the dataset.
# Each object stores the original tokens, original labels, as well as the wordpiece-tokenized versions of those tokens.
# It also stores a map that maps the indexes from the tokens to their position in wordpieces.
# vocab stores info on word_to_ix, wordpiece_to_ix etc, which the Sentence class updates upon creating a sentence.
class Sentence():
	def __init__(self, tokens, labels_triples, labels_entity_types, word_vocab, wordpiece_vocab):
	
		self.tokens = tokens
		self.labels_tr = labels_triples
		self.labels_et = labels_entity_types

		self.mentions_tr = self.get_mentions_vector(self.labels_tr)
		self.mentions_et = self.get_mentions_vector(self.labels_et)
		self.wordpieces, self.token_idxs_to_wp_idxs = self.get_wordpieces(tokens)

		#if len(self.wordpieces) > MAX_SENT_LEN:
		#	logger.debug("A sentence in the dataset exceeds MAX_SENT_LEN (%d): %s" % (MAX_SENT_LEN, " ".join(self.wordpieces)))

		self.wordpiece_labels_tr = self.get_wordpiece_labels(self.wordpieces, self.labels_tr, self.token_idxs_to_wp_idxs)
		self.wordpiece_labels_et = self.get_wordpiece_labels(self.wordpieces, self.labels_et, self.token_idxs_to_wp_idxs)

		# Pad the wordpieces and wordpiece labels so that the DataLoader interprets them correctly.
		self.pad_wordpieces()
		self.pad_wordpiece_labels_tr()
		self.pad_wordpiece_labels_et()
		self.pad_tokens()
		self.pad_labels_tr()
		self.pad_labels_et()
		self.pad_token_map()

		self.wordpiece_mentions_tr = self.get_mentions_vector(self.wordpiece_labels_tr)
		self.wordpiece_mentions_et = self.get_mentions_vector(self.wordpiece_labels_et)

		# Add every word and wordpiece in this sentence to the Vocab object.
		for word in self.tokens:
			word_vocab.add_token(word)
		for wordpiece in self.wordpieces:
			wordpiece_vocab.add_token(wordpiece)

		# Generate the token indexes and wordpiece indexes.
		self.token_indexes = [word_vocab.token_to_ix[token] for token in self.tokens]
		self.wordpiece_indexes = [wordpiece_vocab.token_to_ix[wordpiece] for wordpiece in self.wordpieces]




	
	def get_wordpieces(self, orig_tokens):
		#print orig_to_tok_map, "<<<"
		return tokens_to_wordpieces(orig_tokens)

	# Pad the wordpieces to MAX_SENT_LEN
	def pad_wordpieces(self):
		for x in range(MAX_SENT_LEN - len(self.wordpieces)):
			self.wordpieces.append("[PAD]")

	# Pad the wordpiece_labels to MAX_SENT_LEN
	def pad_wordpiece_labels_tr(self):
		for x in range(MAX_SENT_LEN - len(self.wordpiece_labels_tr)):
			self.wordpiece_labels_tr.append([0] * len(self.wordpiece_labels_tr[0]))

	# Pad the wordpiece_labels to MAX_SENT_LEN
	def pad_wordpiece_labels_et(self):
		for x in range(MAX_SENT_LEN - len(self.wordpiece_labels_et)):
			self.wordpiece_labels_et.append([0] * len(self.wordpiece_labels_et[0]))

	# Pad the tokens to MAX_SENT_LEN
	def pad_tokens(self):
		for x in range(MAX_SENT_LEN - len(self.tokens)):
			self.tokens.append("[PAD]")

	# Pad the labels to MAX_SENT_LEN
	def pad_labels_tr(self):
		for x in range(MAX_SENT_LEN - len(self.labels_tr)):
			self.labels_tr.append([0] * len(self.labels_tr[0]))

	# Pad the labels to MAX_SENT_LEN
	def pad_labels_et(self):
		for x in range(MAX_SENT_LEN - len(self.labels_et)):
			self.labels_et.append([0] * len(self.labels_et[0]))

	# Pad the token to wordpiece map to MAX_SENT_LEN
	def pad_token_map(self):
		for x in range(MAX_SENT_LEN - len(self.token_idxs_to_wp_idxs)):
			self.token_idxs_to_wp_idxs.append(-1)

	# Retrieve the wordpiece labels, which are the same as their corresponding tokens' labels.
	# This is performed using the token_idxs_to_wp_idxs map.
	def get_wordpiece_labels(self, wordpieces, labels, token_idxs_to_wp_idxs):
		wordpiece_labels = []
		padding_labels = [0] * len(labels[0])
		for i, idx in enumerate(token_idxs_to_wp_idxs):

			#for ix in idx:
			#	wordpiece_labels.append(labels[i])
			if i == len(token_idxs_to_wp_idxs) - 1:
				max_idx = len(wordpieces)
			else:
				max_idx = token_idxs_to_wp_idxs[i + 1]
			for x in range(idx, max_idx):
				wordpiece_labels.append(labels[i])

		return [padding_labels] + wordpiece_labels + [padding_labels] # Add 'padding_labels' for the [CLS] and [SEP] wordpieces

	# Retrieve the mentions vector, a list of 0s and 1s, where 1s represent that the token at that index is an entity.
	def get_mentions_vector(self, labels):
		return [1 if 1 in x else 0 for x in labels]

	# Return the data corresponding to this sentence.
	def data(self):
		return self.tokens, self.labels, self.mentions, self.wordpieces, self.wordpiece_labels, self.wordpiece_mentions, self.token_idxs_to_wp_idxs

	# Returns True when this sentence is valid (i.e. its length is <= MAX_SENT_LEN.)
	def is_valid(self):
		return len(self.wordpieces) == MAX_SENT_LEN and len(self.wordpiece_labels_tr) == MAX_SENT_LEN and len(self.wordpiece_labels_et) == MAX_SENT_LEN

	# Print out a summary of the sentence.
	def __repr__(self):
		s =  "Tokens:                  %s\n" % self.tokens
		s += "Labels (triples):        %s\n" % self.labels_tr 
		s += "Labels (entities):       %s\n" % self.labels_et 
		s += "Mentions (tr):           %s\n" % self.mentions_tr 
		s += "Mentions (et):           %s\n" % self.mentions_et
		s += "Wordpieces:              %s\n" % self.wordpieces 
		s += "Wordpiece labels (tr):   %s\n" % self.wordpiece_labels_tr
		s += "Wordpiece labels (et):   %s\n" % self.wordpiece_labels_et 
		s += "Wordpiece mentions (tr): %s\n" % self.wordpiece_mentions_tr
		s += "Wordpiece mentions (et): %s\n" % self.wordpiece_mentions_et
		s += "Token map:               %s\n" % self.token_idxs_to_wp_idxs
		return s



class UnlabeledSentence(Sentence):
	def __init__(self, tokens, word_vocab, wordpiece_vocab):
		self.tokens = tokens
		self.wordpieces, self.token_idxs_to_wp_idxs = self.get_wordpieces(tokens)

		# Pad the wordpieces and wordpiece labels so that the DataLoader interprets them correctly.
		self.pad_wordpieces()
		self.pad_tokens()
		self.pad_token_map()

		# Add every word and wordpiece in this sentence to the Vocab object.
		for word in self.tokens:
			word_vocab.add_token(word)
		for wordpiece in self.wordpieces:
			wordpiece_vocab.add_token(wordpiece)

		# Generate the token indexes and wordpiece indexes.
		self.token_indexes = [word_vocab.token_to_ix[token] for token in self.tokens]
		self.wordpiece_indexes = [wordpiece_vocab.token_to_ix[wordpiece] for wordpiece in self.wordpieces]


	def __repr__(self):
		s =  "Tokens:             %s\n" % self.tokens
		s += "Wordpieces:         %s\n" % self.wordpieces 
		s += "Token map:          %s\n" % self.token_idxs_to_wp_idxs
		return s			

# Expands a label from tail/3 to tail/1/2/3, for example
def expand_labels(labels):
	expanded_labels = set()
	for l in labels:
		if "/" not in l:
			expanded_labels.add(l)
			continue

		t, n = l.split('/')
		for x in range(1, int(n) + 1):
			expanded_labels.add(t + ''.join(['/' + str(y) for y in range(1, x + 1)]))
	return list(expanded_labels)

# Builds the hierarchy given a list of file paths (training and test sets, for example).
def build_hierarchy(filepaths, label_type):
	# Load the hierarchy
	logger.info("Building category hierarchy.")
	hierarchy = CategoryHierarchy()
	for ds_name, filepath in filepaths.items():
		with jsonlines.open(filepath, "r") as reader:
			for i, line in enumerate(reader):
				for m in line['mentions'][label_type]:
					labels = set(m['labels'])
					for l in labels: #expand_labels(labels):						
						hierarchy.add_category(l, ds_name)
	hierarchy.freeze_categories() # Convert the set to a list, and sort it
	logger.info("Category hierarchy contains %d categories." % len(hierarchy))


	return hierarchy

def build_dataset(filepath, hierarchy_tr, hierarchy_et, word_vocab, wordpiece_vocab, ds_name):
	sentences = []
	invalid_sentences_count = 0
	not_enough_labels_sentences_count = 0 # sents containing less than 3 triples
	total_sents = 0
	total_wordpieces = 0
	# Generate the Sentences
	with jsonlines.open(filepath, "r") as reader:
		for line in reader:
			tokens = [w for w in line['tokens']]

			found_h2 = False 
			labels_tr = [[0] * len(hierarchy_tr) for x in range(len(tokens))]
			labels_et = [[0] * len(hierarchy_et) for x in range(len(tokens))]

			for m in line['mentions']['triples']:
				for i in range(m['start'], m['end']):					
					labels_tr[i] = hierarchy_tr.categories2onehot(m['labels']) # expand_labels(m['labels'])
					if 'head/2' in m['labels']:
						found_h2 = True

			for m in line['mentions']['entity_types']:
				for i in range(m['start'], m['end']):
					labels_et[i] = hierarchy_et.categories2onehot(m['labels'])


			sent = Sentence(tokens, labels_tr, labels_et, word_vocab, wordpiece_vocab)
			total_wordpieces += len(sent.wordpieces)
			

			#if not found_h2 and ds_name == "train":
			#	not_enough_labels_sentences_count += 1
			#if found_h2 or ds_name == "dev":
			sentences.append(sent)
		
			if not sent.is_valid():
				invalid_sentences_count += 1
			total_sents += 1
			print("\r%s" % total_sents, end="")
	
	# If any sentences are invalid, log a warning message.
	if invalid_sentences_count > 0:
		logger.warn("%d of %d sentences in the %s dataset were not trimmed due to exceeding MAX_SENT_LEN of %s after wordpiece tokenization." % (invalid_sentences_count, total_sents, ds_name, MAX_SENT_LEN))

	if not_enough_labels_sentences_count > 0:
		logger.warn("%d of %d sentences in the %s dataset were not included due to containing less than 3 triples." % (not_enough_labels_sentences_count, total_sents, ds_name))


	logger.info("Building data loader...")

	if ds_name == "dev":
		print(len(sentences))


	# Construct an EntityTypingDataset object.
	data_x, data_y_tr, data_y_et, data_z_tr, data_z_et, data_i, data_tx, data_ty_tr, data_ty_et, data_tm,  = [], [], [], [], [], [], [], [], [], []
	for i, sent in enumerate(sentences):
		data_x.append(np.asarray(sent.wordpiece_indexes[:cf.MAX_SENT_LEN]))
		data_y_tr.append(np.asarray(sent.wordpiece_labels_tr[:cf.MAX_SENT_LEN]))
		data_y_et.append(np.asarray(sent.wordpiece_labels_et[:cf.MAX_SENT_LEN]))
		data_z_tr.append(np.asarray(sent.wordpiece_mentions_tr[:cf.MAX_SENT_LEN]))
		data_z_et.append(np.asarray(sent.wordpiece_mentions_et[:cf.MAX_SENT_LEN]))
		data_i.append(np.asarray(i))
		data_tx.append(np.asarray(sent.token_indexes[:cf.MAX_SENT_LEN]))
		data_ty_tr.append(np.asarray(sent.labels_tr[:cf.MAX_SENT_LEN]))
		data_ty_et.append(np.asarray(sent.labels_et[:cf.MAX_SENT_LEN]))
		data_tm.append(np.asarray([min(x, cf.MAX_SENT_LEN-1) for x in sent.token_idxs_to_wp_idxs ][:cf.MAX_SENT_LEN])) # Adjust token map to
															# account for overly-long sents
		sys.stdout.write("\r%i / %i" % (i, len(sentences)))
		sys.stdout.flush()
	print("")
	logger.info("Data loader complete.")

	dataset = EntityTypingDataset(data_x, data_y_tr, data_y_et, data_z_tr, data_z_et, data_i, data_tx, data_ty_tr, data_ty_et, data_tm)
	return dataset, sentences, total_wordpieces






def main(opts):

	if len(opts) == 0:
		raise ValueError("Usage: build_data.py <dataset>")
	dataset = opts[0]
	if dataset not in ['cateringServices', 'automotiveEngineering', 'bbn']:
		raise ValueError("Dataset must be either cateringServices, automotiveEngineering, or bbn.")

	cf.load_config(dataset)
	global MAX_SENT_LEN
	MAX_SENT_LEN = cf.MAX_SENT_LEN
	
	dataset_filenames = {
		"train": cf.TRAIN_FILENAME,
		"dev": cf.DEV_FILENAME,
	}

	# 1. Construct the Hierarchy by looking through each dataset for unique labels.
	hierarchy_tr = build_hierarchy(dataset_filenames, 'triples')
	hierarchy_et = build_hierarchy(dataset_filenames, 'entity_types')

	# 2. Construct two empty Vocab objects (one for words, another for wordpieces), which will be populated in step 3.
	word_vocab = Vocab()
	wordpiece_vocab = Vocab()

	logger.info("Hierarchy (tr) contains %d categories unique to the test set." % len(hierarchy_tr.get_categories_unique_to_test_dataset()))
	logger.info("Hierarchy (et) contains %d categories unique to the test set." % len(hierarchy_et.get_categories_unique_to_test_dataset()))

	# 3. Build a data loader for each dataset (train, test).
	data_loaders = {}
	for ds_name, filepath in dataset_filenames.items():
		logger.info("Loading %s dataset from %s." % (ds_name, filepath))
		dataset, sentences, total_wordpieces = build_dataset(filepath, hierarchy_tr, hierarchy_et, word_vocab, wordpiece_vocab, ds_name)
		if ds_name == "dev":
			batch_size = 1
		else:
			batch_size = cf.BATCH_SIZE
		data_loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)
		data_loaders[ds_name] = data_loader
		logger.info("The %s dataset was built successfully." % ds_name)

		logger.info("Dataset contains %i wordpieces (including overly long sentences)." % total_wordpieces)
		if ds_name == "train":
			total_wordpieces_train = total_wordpieces

	BYPASS_SAVING = False
	if BYPASS_SAVING:
		logger.info("Bypassing file saving - training model directly")
		train_without_loading(data_loaders, word_vocab, wordpiece_vocab, hierarchy, total_wordpieces_train)
		return
	

	logger.info("Saving data loaders to file...")

	dutils.save_obj_to_pkl_file(data_loaders, 'data loaders', cf.ASSET_FOLDER + '/data_loaders.pkl')

	logger.info("Saving vocabs and hierarchy to file...")
	dutils.save_obj_to_pkl_file(word_vocab, 'word vocab', cf.ASSET_FOLDER + '/word_vocab.pkl')
	dutils.save_obj_to_pkl_file(wordpiece_vocab, 'wordpiece vocab', cf.ASSET_FOLDER + '/wordpiece_vocab.pkl')
	dutils.save_obj_to_pkl_file(hierarchy_tr, 'hierarchy', cf.ASSET_FOLDER + '/hierarchy_tr.pkl')
	dutils.save_obj_to_pkl_file(hierarchy_et, 'hierarchy', cf.ASSET_FOLDER + '/hierarchy_et.pkl')

	dutils.save_obj_to_pkl_file(total_wordpieces_train, 'total_wordpieces', cf.ASSET_FOLDER + '/total_wordpieces.pkl')

	dutils.save_list_to_file(word_vocab.ix_to_token, 'word vocab', cf.DEBUG_FOLDER + '/word_vocab.txt')
	dutils.save_list_to_file(wordpiece_vocab.ix_to_token, 'wordpiece vocab', cf.DEBUG_FOLDER + '/wordpiece_vocab.txt')


if __name__ == "__main__":
	main(sys.argv[1:])
