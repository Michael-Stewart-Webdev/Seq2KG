
import data_utils as dutils
from config import device
from logger import logger
import torch, json, sys



from data_utils import batch_to_wordpieces, wordpieces_to_bert_embs, build_token_to_wp_mapping, tokens_to_wordpieces


sys.path.append('../evaluation')

from evaluator import evaluate_triples





from colorama import Fore, Back, Style



import random
from nfgec_evaluate import *

from sinusoidal_positional_embeddings import *






from bert_serving.client import BertClient
#from allennlp.modules.elmo import Elmo, batch_to_ids
#from allennlp.commands.elmo import ElmoEmbedder
import numpy as np
import spacy, neuralcoref
from build_data import UnlabeledSentence
from data_utils import EntityTypingDataset
from torch.utils.data import DataLoader
import networkx as nx

# A ModelEvaluator.
# Interacts with a given pre-trained model.
class ModelEvaluator():

	# Create a new model evaluator.
	# model: The model to evaluate
	# dev_loader: The DataLoader containing the dev documents
	# word_vocab: The word vocabulary object
	# wordpiece_vocab: The wordpiece vocabulary object
	# hierarchy: The hierarchy object
	# ground_truth_triples: a list of lists containing the ground truth triples for each document in the dev set
	# cf: The config object
	def __init__(self, model, dev_loader, word_vocab, wordpiece_vocab, hierarchy, ground_truth_triples, cf):
		self.model = model 
		self.dev_loader = dev_loader 
		self.word_vocab = word_vocab 
		self.wordpiece_vocab = wordpiece_vocab
		self.hierarchy = hierarchy		
		self.best_score_and_epoch = [0.0, -1]
		self.ground_truth_triples = ground_truth_triples

		# Set up a new Bert Client, for encoding the wordpieces

		
		self.bc = BertClient()
		self.cf = cf


	# Evaluate a given model via Triples Score over the entire dev corpus.
	# mode = "training" when evaluating during training, and "testing" when evaluating documents directly.
	def evaluate_model(self, epoch, data_loader, mode="training"):

		self.model.zero_grad()
		self.model.eval()
			
		all_txs   = []
		all_tys   = []
		all_preds = []


		self.model.batch_size = 1 # Set the batch size to 1 for evaluation to avoid missing any sents
		for (i, (batch_x, _, _, _, batch_tx, batch_ty, batch_tm)) in enumerate(data_loader):
			
			# 1. Convert the batch_x from wordpiece ids into wordpieces
			#if mode == "training":
			wordpieces = batch_to_wordpieces(batch_x, self.wordpiece_vocab)

			#seq_lens = [len([w for w in doc if w != "[PAD]"]) for doc in wordpieces]

			wordpiece_embs  = wordpieces_to_bert_embs(wordpieces, self.bc).to(device)

			sin_embs = SinusoidalPositionalEmbedding(embedding_dim=self.cf.POSITIONAL_EMB_DIM, padding_idx = 0, left_pad = True)
			sin_embs = sin_embs( torch.ones([wordpiece_embs.size()[0], self.cf.MAX_SENT_LEN])).to(device)
			joined_embs = torch.cat((wordpiece_embs, sin_embs), dim=2)

			# 3. Build the token to wordpiece mapping using batch_tm, built during the build_data stage.
			token_idxs_to_wp_idxs = build_token_to_wp_mapping(batch_tm)


			non_padding_indexes = torch.ByteTensor((batch_tx > 0))

			
			# 4. Retrieve the token predictions for this batch, from the model.
			token_preds = self.model.predict_token_labels(joined_embs, token_idxs_to_wp_idxs)
			

			for batch in token_preds:				
				all_preds.append(batch.int().cpu().numpy().tolist())

			for batch in batch_tx:
				all_txs.append(batch.int().cpu().numpy().tolist())

			if mode == "training":			
				for batch in batch_ty:
					all_tys.append(batch.int().cpu().numpy().tolist())
	
		self.model.batch_size = self.cf.BATCH_SIZE
		tagged_sents = self.get_tagged_sents_from_preds(all_txs, all_tys, all_preds)
		predicted_triples = self.get_triples_from_tagged_sents(tagged_sents)
		logger.info("\n" + self.get_tagged_sent_example(tagged_sents[:1]))	

		def build_true_and_preds(tys, preds):
			true_and_prediction = []
			for b, batch in enumerate(tys):
				for i, row in enumerate(batch):		
					true_cats = self.hierarchy.onehot2categories(tys[b][i])		
					pred_cats = self.hierarchy.onehot2categories(preds[b][i])
					true_and_prediction.append((true_cats, pred_cats))	
			return true_and_prediction	

		if mode == "training":
					
			logger.info("Predicted Triples: ")
			triples_str = ""
			for idx, a in enumerate(predicted_triples):
				for t in a:
					triples_str += "%s %s\n" % (idx, ",".join([" ".join(w) for w in t]))
			logger.info("\n" + triples_str)

			print("")

			# logger.info("Ground truth triples: ")
			
			# triples_str = ""
			# for idx, a in enumerate(self.ground_truth_triples):
			# 	for t in a:
			# 		triples_str += "%s %s\n" % (idx, ", ".join([" ".join(w) for w in t]))
			# logger.info("\n" + triples_str)



			triples_scores = []
			for pt, gt in zip(predicted_triples, self.ground_truth_triples):
				ts = evaluate_triples([[' '.join(t[0]), ' '.join(t[1]), ' '.join(t[2])] for t in pt], [[' '.join(t[0]), ' '.join(t[1]), ' '.join(t[2])] for t in gt])
				triples_scores.append(ts)
			triples_score = sum(triples_scores) / len(triples_scores)
			#triples_score = 0.0
			true_and_predictions = build_true_and_preds(all_tys, all_preds)
			micro_f1 = loose_micro(true_and_predictions)[2]
			logger.info("                  Micro F1: %.4f\t" % (micro_f1))	
			logger.info("                  Triples score: %.4f\t" % (triples_score))	
			return triples_score
		else:
			return predicted_triples

	# Convert the tagged sentence the model produces into triples.
	def get_triples_from_tagged_sents(self, tagged_sents):		
		
		def get_triples(tagged_sents):
			all_triples = []
			for s in tagged_sents:
				label_mapping = {"head": 0, "rel": 1, "tail": 2}
				current_labels = set()
				triples     = [[ [] , [] , [] ] for x in range(10)]
				triple_idxs = [[ [] , [] , [] ] for x in range(10)]
				for word_idx, (word, correct_labels, predicted_labels) in enumerate(s):
					#print(word, predicted_labels, current_labels)
					ls = predicted_labels
					for label in ls:
						label_type, idx = label.split("/")
						idx = int(idx) - 1
						#print(label, current_labels, triples[idx][label_mapping[label_type]])
						if label in current_labels or len(triples[idx][label_mapping[label_type]]) == 0:

							if len(triple_idxs[idx][label_mapping[label_type]]) == 0 or	triple_idxs[idx][label_mapping[label_type]][-1] == word_idx - 1:
									triple_idxs[idx][label_mapping[label_type]].append(word_idx)
									triples[idx][label_mapping[label_type]].append(word)
					
					current_labels = set(predicted_labels)
					
				all_triples.append([t for t in triples if [] not in t])	
			return [t for t in all_triples]	

		predicted_triples = get_triples(tagged_sents)
	
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

	# Iterates through all_txs (the batches/sentences/words), all_preds (batches/sentences/predictions), and all_tys
	# (batches/sentences/correct_labels) to produce a tagged_sents list, in the form of
	# [word, correct_label, prediction]
	def get_tagged_sents_from_preds(self, all_txs, all_tys, all_preds):
		tagged_sents = []
		for b, batch in enumerate(all_txs):
			tagged_sent = []
			for i, token_ix in enumerate(batch):
				if token_ix == 0:
					continue	# Ignore padding tokens

				tagged_sent.append([									\
					self.word_vocab.ix_to_token[token_ix],				\
					[] if all_tys == [] else [l for l in self.hierarchy.onehot2categories(all_tys[b][i]) if "/" in l],	\
					[l for l in self.hierarchy.onehot2categories(all_preds[b][i]) if "/" in l]	\
				])
			tagged_sents.append(tagged_sent)
		return tagged_sents


	# Get an example tagged sentence, returning it as a string.
	# It resembles the following:
	#
	# word_1		Predicted: /other		Actual: /other
	# word_2		Predicted: /person		Actual: /organization
	# ...
	#
	def get_tagged_sent_example(self, tagged_sents):

		# 1. Build a list of tagged_sents, in the form of:
		#    [[word, [pred_1, pred_2], [label_1, label_2]], ...]
		
		# 2. Convert the tagged_sents to a string, which prints nicely
			
		s = ""		
		for tagged_sent in tagged_sents:

			inside_entity = False
			current_labels = []
			current_preds  = []
			current_words  = []
			for tagged_word in tagged_sent:

				is_entity = len(tagged_word[1]) > 0 or len(tagged_word[2]) > 0	
		
				current_labels = tagged_word[1]
				current_preds = tagged_word[2]
				#if (not is_entity and inside_entity) or (is_entity and ((len(current_preds) > 0 or len(current_labels) > 0) and tagged_word[1] != current_labels)):	
				s += "".join(tagged_word[0])[:37].ljust(40)					
				s += "Predicted: "

				if len(current_preds) == 0:
					ps = "%s<No predictions>%s" % (Fore.YELLOW, Style.RESET_ALL)
				else:
					ps = ", ".join(["%s%s%s" % (Fore.GREEN if pred in current_labels else Fore.RED, pred, Style.RESET_ALL) for pred in current_preds])
				
				s += ps.ljust(40)
				s += "Actual: "
				if len(current_labels) == 0:
					s += "%s<No labels>%s" % (Fore.YELLOW, Style.RESET_ALL)
				else:
					s += ", ".join(current_labels)
				s += "\n"

				inside_entity = False
				current_labels = []
				current_preds  = []
				current_words  = []

				# if is_entity:					
				# 	if not inside_entity:
				# 		inside_entity = True
				# 		current_labels = tagged_word[1]
				# 		current_preds = tagged_word[2]

				# 	current_words.append(tagged_word[0])
			s += "\n"
			
		return s

	# Save the best model to the best model directory, and save a small json file with some details (epoch, f1 score).
	def save_best_model(self, triples_score, epoch):
		logger.info("Saving model to %s." % self.cf.BEST_MODEL_FILENAME)
		torch.save(self.model.state_dict(), self.cf.BEST_MODEL_FILENAME)

		logger.info("Saving model details to %s." % self.cf.BEST_MODEL_JSON_FILENAME)
		model_details = {
			"epoch": epoch,
			"triples_score": triples_score
		}
		with open(self.cf.BEST_MODEL_JSON_FILENAME, 'w') as f:
			json.dump(model_details, f)

	# Determine whether the given score is better than the best score so far.
	def is_new_best_triples_score(self, score):
		return score > self.best_score_and_epoch[0]

	# Determine whether there has been no improvement to score over the past n epochs.
	def no_improvement_in_n_epochs(self, n, epoch):
		return epoch >= 15 and self.best_score_and_epoch[0] > 0 and (epoch - self.best_score_and_epoch[1] >= n)

	# Evaluate the model every n epochs.
	def evaluate_every_n_epochs(self, n, epoch):		
		if epoch % n == 0 or epoch == 1000:
			score = self.evaluate_model(epoch, self.dev_loader, mode="training")

			if self.is_new_best_triples_score(score):
				self.best_score_and_epoch = [score, epoch]
				logger.info("New best average Triples Score achieved!        (%s%.4f%s)" % (Fore.YELLOW, score, Style.RESET_ALL))
				self.save_best_model(score, epoch)
			elif self.no_improvement_in_n_epochs(self.cf.STOP_CONDITION, epoch):#:self.cf.STOP_CONDITION):
				logger.info("No improvement to Triples Score in past %d epochs. Stopping early." % self.cf.STOP_CONDITION)
				logger.info("Best F1 Score: %.4f" % self.best_score_and_epoch[0])
				sys.exit(0)


# Create an evaluator for a given dataset.
# 

class TrainedEndToEndModel():

	# Create a new TrainedEndToEndModel.
	# dataset: The name of the dataset (e.g. 'culinaryServices')
	# cf: The config object
	def __init__(self, dataset, cf):

		from model import E2EETModel
		from bert_serving.client import BertClient
		import jsonlines

		logger.info("Loading files...")

		data_loaders = dutils.load_obj_from_pkl_file('data loaders', cf.ASSET_FOLDER + '/data_loaders.pkl')
		# Note: the word and wordpiece vocab are stored as attributes so that they may be expanded
		# if necessary during evaluation (if a new word appears)
		self.word_vocab = dutils.load_obj_from_pkl_file('word vocab', cf.ASSET_FOLDER + '/word_vocab.pkl')
		self.wordpiece_vocab = dutils.load_obj_from_pkl_file('wordpiece vocab', cf.ASSET_FOLDER + '/wordpiece_vocab.pkl')
		hierarchy = dutils.load_obj_from_pkl_file('hierarchy', cf.ASSET_FOLDER + '/hierarchy.pkl')
		total_wordpieces = dutils.load_obj_from_pkl_file('total wordpieces', cf.ASSET_FOLDER + '/total_wordpieces.pkl')
		
		# Initialise the coref pipeline for use in end-user evaluation
		self.nlp = spacy.load('en')
		self.coref = neuralcoref.NeuralCoref(self.nlp.vocab)
		self.nlp.add_pipe(self.coref, name='neuralcoref')
		
		logger.info("Building model.")
		model = E2EETModel(	embedding_dim = cf.EMBEDDING_DIM + cf.POSITIONAL_EMB_DIM,
							hidden_dim = cf.HIDDEN_DIM,
							vocab_size = len(self.wordpiece_vocab),
							label_size = len(hierarchy),
							total_wordpieces = total_wordpieces,
							category_counts = hierarchy.get_train_category_counts(),
							hierarchy_matrix = hierarchy.hierarchy_matrix,
							max_seq_len = cf.MAX_SENT_LEN,
							batch_size = 1)
		model.cuda()

		model.load_state_dict(torch.load(cf.BEST_MODEL_FILENAME))

		self.modelEvaluator = ModelEvaluator(model, None, self.word_vocab, self.wordpiece_vocab, hierarchy, None, cf)
		self.cf = cf


	# Evaluate the model on a list of strings and return the list of lists of triples predicted by the model.
	def triples_from_docs(self, docs):

		if self.cf.DATASET != "bbn":


			logger.info("Evaluating triples from %d docs..." % len(docs))
			docs = [self.nlp(doc_str) for doc_str in docs]
			spacy_tokens_all = [[str(w) for w in doc] for doc in docs]

			# Perform coref resolution on each document TODO: Move to separate file
			spacy_tokens_corefed = []
			for i, spacy_tokens in enumerate(spacy_tokens_all):			
				offset = 0
				if(docs[i]._.has_coref):
					for c in docs[i]._.coref_clusters:
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
				spacy_tokens_corefed.append(spacy_tokens)

			# Split into sentences
			docs = []
			for doc in spacy_tokens_corefed:	
				current_sent = []
				docs.append([])		
				for word in doc:
					if word == "." and len(current_sent) > 0:
						docs[-1].append(current_sent)
						current_sent = []
						continue
					current_sent.append(word)
				if len(current_sent) > 0:
					docs[-1].append(current_sent)
		else:
			docs = [[doc.split()] for doc in docs]
		
		# Create UnlabeledSentence objects and use them to create a DataLoader.
		evaluation_docs = []
		for doc in docs:
			evaluation_docs.append([])
			for sent in doc:
				unlabeled_sent = UnlabeledSentence(sent, self.word_vocab, self.wordpiece_vocab)
				evaluation_docs[-1].append(unlabeled_sent)

		# Build the data loader and evaluate, per document.
		triples = [] # Triples is a list of list, where each outer element contains a list of triples for 
					 # each document.

		logger.info("Constructing dataloader for documents...")

		for doc_idx, doc in enumerate(evaluation_docs):

			data_x, data_y, data_z, data_i, data_tx, data_ty, data_tm,  = [], [], [], [], [], [], []
			for i, sent in enumerate(doc):
				data_x.append(np.asarray(sent.wordpiece_indexes)[:self.cf.MAX_SENT_LEN])
				data_y.append(np.asarray([]))
				data_z.append(np.asarray([]))
				data_i.append(np.asarray(i))
				data_tx.append(np.asarray(sent.token_indexes)[:self.cf.MAX_SENT_LEN])
				data_ty.append(np.asarray([]))
				data_tm.append(np.asarray([min(x, self.cf.MAX_SENT_LEN-1) for x in sent.token_idxs_to_wp_idxs ][:self.cf.MAX_SENT_LEN]))
		
			dataset = EntityTypingDataset(data_x, data_y, data_z, data_i, data_tx, data_ty, data_tm)
			data_loader = DataLoader(dataset, batch_size=1, pin_memory=True)

			print("\r%d / %d" % (doc_idx, len(evaluation_docs)), end="")
			ts = self.modelEvaluator.evaluate_model(1, data_loader, mode="testing")
			triples.append(sum(ts, [])) # flatten back into document-level
		print("")


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

	evaluator = TrainedEndToEndModel(dataset, cf)
	print(evaluator.triples_from_docs(["KFC is a restaurant that serves fried chicken. Barrack ate at the restaurant.", "The restaurateur and Santa Cruz resident behind three Michelin - starred Manresa in Los Gatos has signed on to the endeavor , where he ’ ll open an as - yet unnamed business"]))		

if __name__ == "__main__":

	main(sys.argv[1:])


