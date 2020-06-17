# sents_to_triples.py by Michael Stewart
# Parses a Redcoat document and returns a list of triples.

import sys, jsonlines, random
import jsonlines
from build_triples import build_triples, get_coreffed_docs_from_doc
from colorama import Fore, Back, Style

sys.path.append('../candidate_extraction')

from triples_from_text import extract_triples
from bert_serving.client import BertClient


sys.path.append('../end_to_end_model/bert')
import tokenization

from scipy.spatial.distance import cosine 
import spacy, neuralcoref


tokenizer = tokenization.FullTokenizer(
    vocab_file='../end_to_end_model/bert/vocab.txt', do_lower_case=False)

USE_EMB_DISTANCE = False

# Transforms a list of original tokens into a list of wordpieces using the Bert tokenizer.
# Returns two lists:
# - bert_tokens, the wordpieces corresponding to orig_tokens,
# - orig_to_tok_map, which maps the indexes from orig_tokens to their positions in bert_tokens.
#   for example, if orig_tokens = ["hi", "michael"], and bert_tokens is ["[CLS]", "hi", "mich", "##ael", "[SEP]"],
#	then orig_to_tok_map becomes [1, 2].
def tokens_to_wordpieces(orig_tokens):
	bert_tokens = []
	orig_to_tok_map = []
	bert_tokens.append("[CLS]")
	for orig_token in orig_tokens:
		word_pieces = tokenizer.tokenize(orig_token)
		orig_to_tok_map.append(len(bert_tokens))# + x for x in range(len(word_pieces))])
		bert_tokens.extend(word_pieces)
	bert_tokens.append("[SEP]")
	return bert_tokens, orig_to_tok_map

# Return a list of triples that are present in a Redcoat document.
def get_redcoat_triples(doc):
	mentions = doc['mentions']
	if 'triples' in doc['mentions']:
		mentions = doc['mentions']['triples']
	triples = [[None, None, None] for x in range(10)]
	ordering = {"head": 0, "rel": 1, "tail": 2}



	for m in mentions:
		start = m['start']
		end = m['end']
		#labels = [l.split("/")[1] for l in m['labels'] if "/" in l]
		labels = m['labels']
		tokens = doc['tokens'][start:end]

		for l in labels:
			if "/" not in l:
				continue			
			t, n = l.split("/")	
			triples[int(n) - 1][ordering[t]] = " ".join(tokens).rstrip('.')
	return [t for t in triples if None not in t]

def extract_triples_michael(doc_str):
	return build_triples([doc_str], verbose=False)

# Parse the train and dev datasets (in the form of Redcoat annotated documents) and return two lists:
# - documents['train'], the documents from the training set
# - documents['dev'], the documents from the dev set
def get_documents_from_redcoat_annotations(dataset, dataset_name):
	documents = {}
	documents[dataset_name] = []
	for doc in dataset:
		# Raise an exception if a document does not have any mentions.
		if len(doc['mentions']) == 0:
			raise Exception("All sentences passed to get_documents_from_redcoat_annotations should have at least one mention.")
			exit()
		doc_str = " ".join(doc['tokens'])
		documents[dataset_name].append(doc_str)
	return documents[dataset_name]


# Parse the train and dev datasets (in the form of Redcoat annotated documents) and return three lists:
# - triples['train'], the triples extracted from the training set
# - triples['dev'], the triples extracted from the dev set
# - ground_truth_triples, the list of ground_truth_triples in the dev set.
def create_triples_from_redcoat_annotations(dataset, dataset_name, extraction_model='majiga'):
	print("Extracting triples from %s set..." % dataset_name, end="")	
	unlabelled_docs = 0
	
	num_correct = 0
	num_total = 0
	if USE_EMB_DISTANCE:
		bc = BertClient(port=6555, port_out=6556)
	
	triples = {}
	triples_per_doc = []

	if extraction_model=="michael":
		nlp = spacy.load('en')
		coref = neuralcoref.NeuralCoref(nlp.vocab)
		nlp.add_pipe(coref, name='neuralcoref')


	doc_idx = 0
	triples[dataset_name] = []
	for doc in dataset:		

		num_hits_this_doc = 0

		# Raise an exception if a document does not have any mentions.
		if len(doc['mentions']) == 0:				
			raise Exception("All documents passed to create_triples should have at least one mention.")
			exit()

		doc_str = " ".join(doc['tokens'])		
				
		print(doc_str)
		ground_truth_triples = get_redcoat_triples(doc)
		print(ground_truth_triples)

		#if dataset_name == "dev":
		#	all_ground_truth_triples += [[doc_idx] + gt + [1] for gt in ground_truth_triples]
		#print("Document: ")
		#print(doc_str)
		#print("")
		if USE_EMB_DISTANCE:
			ground_truth_embs = []
			print("Ground truths: ")
			for gt in ground_truth_triples:
				if not any([len(p) == 0 for p in gt]): # Ignore ground truths with empty strings
					ground_truth_embs.append([bc.encode([p]) for p in gt])

					print(gt, Style.RESET_ALL)
			print("")
		else:
			joined_gt_set = set(["�".join(t) for t in ground_truth_triples])
		

		#print("extracting...")
		if extraction_model == 'majiga':			
			if not(doc_str.endswith('.')):
				doc_str = doc_str + '.' # required for extract_triples to work
			extracted_triples = extract_triples(doc_str)
		elif extraction_model == 'michael':
			doc_str = get_coreffed_docs_from_doc(nlp(doc_str))

			extracted_triples = extract_triples_michael(doc_str) # Using michael's here because Majiga's 
				# triple extraction code operates on documents, not sentences
				# (this function shouldn't really be called with 'sentences' as a parameter unless it is
				# being used to generate data for the end-to-end model ... I need to update the code to
				# reflect this)
			
		#print("Document (after coref): ")
		#print(doc_str)
		#print("")
		
		#print(doc_str)
		#joined_gt_set = set(["�".join(t) for t in ground_truth_triples])
		
		labelled_triples = []

		for t in extracted_triples:

			#if granularity == 'sentences':
			#	break
			if extraction_model == 'michael':
				t = t[1:] # michael's model adds doc_idx at the start of each triple
			

			if USE_EMB_DISTANCE:
				correct = 0

				
				if not any([len(p) == 0 for p in t]): # Triples with empty strings are automatically incorrect					
					t_embs = [bc.encode([p]) for p in t]
					for i, gte in enumerate(ground_truth_embs):
						dists = []
						for part in [0, 1, 2]:						
							dists.append(cosine(t_embs[part], gte[part]))					
						#dist = sum(dists) / 3.0
						if all([d < 0.075 for d in dists]):# < 0.06: #0.065:
							correct = 1
							num_hits_this_doc += 1
	

				if correct == 1:
					col = Fore.GREEN
				else:
					col = ""
				#print(col, t, correct, Style.RESET_ALL)

				#joined_triple = "�".join(t)
			
				#correct = int(joined_triple in joined_gt_set)
				#labelled_triples.append(t + [correct])
				labelled_triples.append([doc_idx] + t + [correct])
			else:
				joined_triple = "�".join(t)
				correct = int(joined_triple in joined_gt_set)
				labelled_triples.append([doc_idx] + t + [correct])	
				num_hits_this_doc += correct				


		for t in labelled_triples:
			if t[-1] == 1:
				num_correct += 1
			#if t[-1] == 0:	# Add correct triples to the train set only if it is not in the ground truth set.			
			#if dataset_name == "dev" or t[-1] == 0:	# Add correct triples to the train set only if it is not in the ground truth set.			
			if t[-1] == 0:
				triples[dataset_name].append(t)					

		num_total += len(ground_truth_triples)
		
		# Only add ground truth triples to the training set, but not to the validation set
		# This ensures the predictions of the filtering model reflect what it would predict given the triples from the
		# candidate extraction model
		#if dataset_name == "train": 
		triples[dataset_name] += [[doc_idx] + gt + [1] for gt in ground_truth_triples]
		

		triples_per_doc.append(num_hits_this_doc)
		doc_idx += 1
		print("\r" + Fore.YELLOW + "Extracting triples from %s set... %s / %s         %s / %s                Average: %.2f               %s\n"
			 % (dataset_name, doc_idx, len(dataset), num_hits_this_doc, len(triples[dataset_name]), sum(triples_per_doc) / len(triples_per_doc), Style.RESET_ALL), end="")	

		print()
		if not USE_EMB_DISTANCE:
			print("\n%d / %d correct (%.4f%s) (%d total)" % (num_correct, num_total, num_correct / num_total * 100, "%", len(triples[dataset_name])))
		#print("Average triples per doc: %.2f" % (sum(triples_per_doc) / len(triples_per_doc)))
	return triples[dataset_name]#, all_ground_truth_triples

	
	


# Only add ground truth triples to the training set, but not to the validation set
										# This ensures the predictions of the filtering model reflect what it would predict given the triples from the
										# candidate extraction model
# if __name__ == "__main__":
# 	train_data, dev_data = get_redcoat_data_already_split()
# 	create_dataset(train_data, 'train')
# 	create_dataset(dev_data, 'dev')