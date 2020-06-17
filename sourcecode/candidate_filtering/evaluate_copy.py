import data_utils as dutils
from config import device
from logger import logger
import torch, json, sys
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score, classification_report, accuracy_score, recall_score, precision_score

from colorama import Fore, Back, Style

sys.path.append('../evaluation')
from evaluator import evaluate_triples

import random
torch.manual_seed(123)
import networkx as nx
import numpy as np

sys.path.append('../redcoat_parser')
from build_triples import build_triples
sys.path.append('../candidate_extraction')
from triples_from_text import extract_triples
import pandas as pd
from build_data import build_dataset
import spacy, neuralcoref

TOP_N_TRIPLES = 10 # Return only the first TOP_N_TRIPLES, sorted by highest model confidence.

class ModelEvaluator():

	def __init__(self, model, dev_loader, evaluation_dataset, ground_truth_triples, cf):
		self.model = model 
		self.dev_loader = dev_loader 
		self.evaluation_dataset = evaluation_dataset
		self.ground_truth_triples = ground_truth_triples
		self.best_score_and_epoch = [0.0, -1]
		self.cf = cf

	# Evaluate a given model via Triples Score over the entire test corpus.
	def evaluate_model(self, epoch, data_loader, mode="training"):

		self.model.zero_grad()
		self.model.eval()
		
		all_truths   = []
		all_preds = []
	

		#print("")
		#print("%s %s %s %s %s" % ("Head".ljust(35), "Relation".ljust(35), "Tail".ljust(35), "Pred    ", "Label"))
		#print("=" * 125)

		#random_evaluate_index = random.randint(0, len(self.data_loader)-1)
		all_predicted_triples = []
		preds_list = []
		
		doc_preds  = {}  # A dict of lists, where each inner list is the truth values for all triples in a document
		doc_confidences  = {}

		for (i, (batch_idx, batch_doc_idx, batch_d, batch_h, batch_r, batch_t, batch_y)) in enumerate(data_loader):

			# 1. Place each component onto CUDA
			batch_d = batch_d.to(device) # document embedding
			batch_h = batch_h.to(device) # head embedding
			batch_r = batch_r.to(device) # relation embedding
			batch_t = batch_t.to(device) # tail embedding
			batch_y = batch_y.float().to(device) # correct labels (0 or 1)
			
			# 2. Retrieve the label predictions for this batch, from the model.
			preds, confidences = self.model.predict_labels(batch_d, batch_h, batch_r, batch_t)


			# 3. Append the ground truth labels and predicted labels to the list
			batch_doc_idx = batch_doc_idx.int().cpu().numpy().tolist()
			batch_y = batch_y.int().cpu().numpy().tolist()
			preds = preds.int().cpu().numpy().tolist()
			confidences = confidences.float().detach().cpu().numpy().tolist()

			all_truths += batch_y
			#all_preds  += preds

			# 4. Build a list of truth values for each document, for example:
			# doc_truths = {
			#		0: [1, 1, 1, 0, 0, 0],
			# 		1: [1, 0, 0, 0, 0, 0]
			#	}
			#	.. and so on.
			
			batch_idx = batch_idx.int().cpu().numpy().tolist()
			for j, doc_idx in enumerate(batch_doc_idx):	

				if doc_idx not in doc_preds:
					doc_preds[doc_idx] = []				
				doc_preds[doc_idx].append(preds[j])

				if doc_idx not in doc_confidences:
					doc_confidences[doc_idx] = []				
				doc_confidences[doc_idx].append(confidences[j])

			#if i == random_evaluate_index:
			#self.get_labeled_example(batch_idx, preds, batch_y)

		
		# Iterate over the dictionary mapping doc_idx to a list of confidences and obtain a list of predictions
		# based on the TOP_N_TRIPLES variable.
		# This essentially means the model predicts up to TOP_N_TRIPLES, sorted in descending order of confidence.
		# The model can predict less than N triples if there are not 10 predictions whose confidence exceeds the
		# threshold defined in model.py.
		for (doc_idx, confidences) in doc_confidences.items():
			preds = np.asarray(doc_preds[doc_idx])
			confidences = np.asarray(confidences)			
			assert len(confidences) == len(preds)
			sort_indexes = np.argsort(np.argsort(confidences)[::-1]) # Argsort twice to obtain the ranks			
			mask = (sort_indexes < TOP_N_TRIPLES).astype(int)			
			preds_filtered =  mask * preds
			all_preds  += preds_filtered.tolist()

	
		
		all_predicted_triples = self.get_predicted_triples_list(all_preds)
		

	

		if mode == 'training':
			logger.info("Predicted triples:")
			self.print_triples_list(all_predicted_triples)

			logger.info("Ground truth triples:")
			self.print_triples_list(self.ground_truth_triples)


			triples_scores = []
			for pt, gt in zip(all_predicted_triples, self.ground_truth_triples):
				ts = evaluate_triples([[' '.join(t[0]), ' '.join(t[1]), ' '.join(t[2])] for t in pt], [[' '.join(t[0]), ' '.join(t[1]), ' '.join(t[2])] for t in gt])
				triples_scores.append(ts)
			triples_score = sum(triples_scores) / len(triples_scores)

			acc = accuracy_score(all_truths, all_preds)
			f1 = f1_score(all_truths, all_preds)
			p = precision_score(all_truths, all_preds)
			r = recall_score(all_truths, all_preds)

			#triples_score = self.get_triples_score(all_predicted_triples, self.ground_truth_triples)

			logger.info("Triples score: %.4f" % triples_score)
			logger.info("Accuracy: %.4f" % acc)
			logger.info("Precision: %.4f" % p)
			logger.info("Recall: %.4f" % r)
			logger.info("F1 Score: %.4f" % f1)
			#logger.info("Triples Score: %.4f" % triples_score)
			return triples_score
		elif mode == 'testing':
			return all_predicted_triples


	def print_triples_list(self, triples):
		s = ""
		for i, doc in enumerate(triples):
			for (head, rel, tail) in doc:
				s += "%s %s %s %s\n" % (str(i).ljust(3), " ".join(head)[:29].ljust(35), " ".join(rel[:29]).ljust(35), " ".join(tail[:29]).ljust(35))
		logger.info("\n%s\n" % s)

	def get_predicted_triples_list(self, preds):

		def check_row_is_valid(row):
			return not (pd.isnull(getattr(row, 's1')) or pd.isnull(getattr(row, 'r')) or pd.isnull(getattr(row, 's2')) or pd.isnull(getattr(row, 'label')))


		num_docs = self.evaluation_dataset['index'].max()
		predicted_triples = [ [] for x in range(num_docs)]

		i = 0
		current_index = -1	
		for row in self.evaluation_dataset.itertuples():
			if not check_row_is_valid(row): # Ignore invalid rows, same as how build_data.py ignores them when building
				continue

			print(row, i, len(preds))

			index = int(getattr(row, 'index'))
			head = getattr(row, 's1')
			relation = getattr(row, 'r')
			tail = getattr(row, 's2')			
			if index > current_index:
				predicted_triples.append([])
				current_index = index
			if preds[i] == 1:
				predicted_triples[current_index].append([head.split(), relation.split(), tail.split()])	
			i += 1	

		return predicted_triples


	# Evaluates the quality of the predicted triples against the ground truth triples and returns a score.
	def get_triples_score(self, predicted_triples, ground_truth_triples):

		def jaccard_similarity(list1, list2):
			s1 = set(list1)
			s2 = set(list2)
			return len(s1.intersection(s2)) / len(s1.union(s2))

		# # Returns True if two nodes are considered the 'same', i.e. greater than 50% jaccard similarity.
		# def same_nodes(n1, n2):
		# 	t1 = n1['tokens']
		# 	t2 = n2['tokens']
		# 	#print(t1, t2, jaccard_similarity(t1, t2))
		# 	return jaccard_similarity(t1, t2) >= 0.5

		def create_graph(triples):
			G = nx.DiGraph()
			G.add_edges_from( [(" ".join(t[0]), " ".join(t[2])) for t in triples] )
			for node in G.nodes:
				G.nodes[node]['tokens'] = node.split()
			return G

		assert(len(predicted_triples) == len(ground_truth_triples))
		
		scores = []
		for i in range(len(ground_truth_triples)): # Iterate over each document

			G_pred  = create_graph(predicted_triples[i])
			G_truth = create_graph(ground_truth_triples[i])

			scores.append(nx.graph_edit_distance(G_pred, G_truth))#, same_nodes))

		return sum(scores) / len(scores)

	# Display the triples for a given sentence index.
	# Correctly-predicted triples are in green, while incorrect ones are in red.
	def get_labeled_example(self, idxs, preds, batch_y):
		idxs = idxs.tolist()
		
		# Read each row from the evaluation dataset file (dev.csv) so that we know what the head/rels/tails were
		# (the data loader only contains embeddings, not strings)
		for i, idx in enumerate(idxs):
			row = self.evaluation_dataset.loc[idx]
			head = getattr(row, 's1')
			relation = getattr(row, 'r')
			tail = getattr(row, 's2')
			pred = "True" if preds[i].cpu().numpy() == 1 else "False"
			truth = "True" if batch_y[i].cpu().numpy() == 1 else "False"
			color = Fore.GREEN if pred == truth else Fore.RED
			if pred == "True" or truth == "True":
				print("%s%s %s %s %s %s%s" % (color, head[:29].ljust(35), relation[:29].ljust(35), tail[:29].ljust(35), pred.ljust(8), truth, Style.RESET_ALL))
		
		return None

	# Save the best model to the best model directory, and save a small json file with some details (epoch, f1 score).
	def save_best_model(self, score, epoch):
		logger.info("Saving model to %s." % self.cf.BEST_MODEL_FILENAME)
		for name, p in self.model.named_parameters():
			print(name, p)
		torch.save(self.model.state_dict(), self.cf.BEST_MODEL_FILENAME)

		logger.info("Saving model details to %s." % self.cf.BEST_MODEL_JSON_FILENAME)
		model_details = {
			"epoch": epoch,
			"score": score
		}
		with open(self.cf.BEST_MODEL_JSON_FILENAME, 'w') as f:
			json.dump(model_details, f)

	# Determine whether the given score is better than the best score so far.
	def is_new_best_score(self, score):
		return score > self.best_score_and_epoch[0]

	# Determine whether there has been no improvement to score over the past n epochs.
	def no_improvement_in_n_epochs(self, n, epoch):
		return epoch - self.best_score_and_epoch[1] >= n

	# Evaluate the model every n epochs.
	def evaluate_every_n_epochs(self, n, epoch):		
		if epoch % n == 0 or epoch == self.cf.MAX_EPOCHS:
			score = self.evaluate_model(epoch, self.dev_loader, mode="training")

			if self.is_new_best_score(score):
				self.best_score_and_epoch = [score, epoch]
				logger.info("New best average Triples Score achieved!        (%s%.4f%s)" % (Fore.YELLOW, score, Style.RESET_ALL))
				self.save_best_model(score, epoch)
			elif self.no_improvement_in_n_epochs(self.cf.STOP_CONDITION, epoch):#:cf.STOP_CONDITION):
				logger.info("No improvement to F1 Score in past %d epochs. Stopping early." % self.cf.STOP_CONDITION)
				logger.info("Best F1 Score: %.4f" % self.best_score_and_epoch[0])
				sys.exit(0)


class TrainedFilteringModel():

	# Create a new TrainedEndToEndModel.
	# dataset: The name of the dataset (e.g. 'culinaryServices')
	# cf: The config object
	def __init__(self, dataset, cf):

		from model import CandidateFilteringModel
		from bert_serving.client import BertClient
		import jsonlines
		

		logger.info("Loading files...")

		#data_loader_train = dutils.load_obj_from_pkl_file('data loader (train)', cf.ASSET_FOLDER + '/data_loader_train.pkl')
		#data_loader_dev   = dutils.load_obj_from_pkl_file('data loader (dev)', cf.ASSET_FOLDER + '/data_loader_dev.pkl')

		logger.info("Building model.")
		model = CandidateFilteringModel(
					embedding_dim = cf.EMBEDDING_DIM,
					hidden_dim = cf.HIDDEN_DIM,
					)
		model.cuda()


		model.load_state_dict(torch.load(cf.BEST_MODEL_FILENAME))

		model.eval()

		self.modelEvaluator = ModelEvaluator(model, None, None, None, cf)
		self.cf = cf

		# Initialise the coref pipeline for use in end-user evaluation
		self.nlp = spacy.load('en')
		self.coref = neuralcoref.NeuralCoref(self.nlp.vocab)
		self.nlp.add_pipe(self.coref, name='neuralcoref')


	# Evaluate the model on a list of strings and return the list of lists of triples predicted by the model.
	def get_coreffed_sents_from_doc(self, doc):
		doc = self.nlp(doc)
		spacy_tokens = [str(w) for w in doc]

		# Perform coref resolution on each document TODO: Move to separate file
			
		offset = 0
		if(doc._.has_coref):
			for c in doc._.coref_clusters:
				for m in c.mentions:
					
					#print(m, m.start, m.end, doc[m.start:m.end], m._.coref_cluster.main)
					orig_len = m.end - m.start
					main_coref = [str(w) for w in m._.coref_cluster.main]

					if len(main_coref) > 3:	# Ignore long coreference mentions
						continue
					#print (main_coref, m._.coref_cluster.main.start)
					new_len = len(main_coref)

					spacy_tokens = spacy_tokens[:m.start + offset] + main_coref + spacy_tokens[m.end + offset:]
					offset += new_len - orig_len

		
		sents = []
		current_sent = []
		for word in spacy_tokens:
			if word == "." and len(current_sent) > 0:
				sents.append(' '.join(current_sent))
				current_sent = []
				continue
			current_sent.append(word)
		if len(current_sent) > 0:
			sents.append(' '.join(current_sent))
		


		return sents

	# Evaluate the model on a list of strings and return the list of lists of triples predicted by the model.
	def triples_from_docs(self, docs):
		triples = []
		
		logger.info("Extracting triples from docs...")

	
		# To use michael's triple extractor:
		all_extracted_triples = []
		for doc_idx, d in enumerate(docs):

			sents = self.get_coreffed_sents_from_doc(d)
			
			for t in build_triples(sents, add_doc_idx=False, verbose=False):
				all_extracted_triples += [[doc_idx] + t + [0]]
			print ("\r%d / %d complete. (%d triples generated)" % ((doc_idx + 1), len(docs), len(all_extracted_triples)), end="" )
		print()



		# # To use Majiga's triple extractor:
		# all_extracted_triples = []
		# for doc_idx, doc in enumerate(docs):
		# 	if not doc.endswith('.'): # triples code requires full stops at the end
		# 		doc = doc + '.'
		# 	extracted_triples = extract_triples(doc)	
		# 	all_extracted_triples += [[doc_idx] + t + [0] for t in extracted_triples] # Add a 'label' for each document so it works with the build_dataset function
		# 	print("\r%d / %d" % (doc_idx, len(docs)), end="")
		# print()

		df = pd.DataFrame(data=all_extracted_triples, columns=['index', 's1', 'r', 's2', 'label'])

		dataset = build_dataset(df, 'evaluation', docs)
		data_loader = DataLoader(dataset, batch_size=1, pin_memory=True)			
		self.modelEvaluator.evaluation_dataset = df
		triples = self.modelEvaluator.evaluate_model(1, data_loader, mode='testing')


		# Clean a string by removing leading or trailing non-alpha/numeric characters
		def clean(s):
			start = -1
			end = -1
			for i, c in enumerate(s):
				if c.isalpha() or c.isnumeric():
					start = i
					break
			for i, c in enumerate(s[::-1]):
				if c.isalpha() or c.isnumeric():
					end = len(s) - i
					break
			return s[start:end]

		# clean the triples
		clean_triples = []
		for doc in triples:
			clean_triples.append([])
			for i, t in enumerate(doc):
				cleaned_triple = [clean(t[0]), clean(t[1]), clean(t[2])]
				if not any([len(x) == 0 for x in cleaned_triple]):
					clean_triples[-1].append(cleaned_triple)

		


		return clean_triples



def main(opts):
	if len(opts) == 0:
		raise ValueError("Usage: evaluate.py <dataset>")
	dataset = opts[0]
	if dataset not in ['cateringServices', 'automotiveEngineering']:
		raise ValueError("Dataset must be either cateringServices or automotiveEngineering.")

	from config import Config
	cf = Config()
	cf.load_config(dataset)

	evaluator = TrainedFilteringModel(dataset, cf)
	triples = evaluator.triples_from_docs(["KFC is a restaurant that serves fried chicken . Barrack ate at the restaurant.", "The restaurateur and Santa Cruz resident behind three Michelin - starred Manresa in Los Gatos has signed on to the endeavor , where he ’ ll open an as - yet unnamed business"])

	print(triples)
if __name__ == "__main__":

	main(sys.argv[1:])
		
