import csv, sys, random, os
import networkx as nx
import gmatch4py as gm
import numpy as np
from scipy.spatial.distance import cosine
from bert_serving.client import BertClient

N_RANDOM_WALKS = 10


random.seed(123)

models = ['candidate_extraction', 'candidate_filtering',  'end_to_end_model']
industries = ['cateringServices', 'automotiveEngineering']

models = ['candidate_filtering']
industries = ['bbn']
#industries = ['bbn', 'cateringServices', 'automotiveEngineering']

bc = BertClient(port=6555, port_out=6556)

def parse_file(reader, num_docs):
	triples = [ [] for x in range(num_docs) ]

	for i, row in enumerate(reader):
		if i == 0: continue
		index = int(row[0])
		triples[index].append(row[1:])
	return triples

# Evaluates the quality of the predicted triples against the ground truth triples and returns a score.
def evaluate_triples(predicted_triples, ground_truth_triples):

	def jaccard_similarity(list1, list2):
		s1 = set(list1)
		s2 = set(list2)
		return len(s1.intersection(s2)) / len(s1.union(s2))

	def create_graph(triples):
		G = nx.DiGraph()
		G.add_edges_from( [(t[0], t[2]) for t in triples] )

		#G.add_edges_from( [(" ".join(t[0]), " ".join(t[2])) for t in triples] )
		#for node in G.nodes:
		#	G.nodes[node]['tokens'] = node.split()
		return G

	G_pred  = create_graph(predicted_triples)
	G_truth = create_graph(ground_truth_triples)

	# x = nx.optimize_edit_paths(G_pred, G_truth)
	
	# costs = []
	# for _, _, cost in x:
	# 	costs.append(cost)

	#score = nx.graph_edit_distance(G_pred, G_truth)#, same_nodes))
	#return score

	if False:
		ged = gm.GraphEditDistance(1,1,1,1)

		result = ged.compare([G_pred, G_truth], None)
		result = ged.similarity(result)

		e1, e2 = result[0][1], result[1][0]
		if np.isinf(e1): # inf means there is no similarity at all because one graph is empty
			e1 = 0
		if np.isinf(e2):
			e2 = 0


	def get_edge_labels(triples):
		edge_labels = {}
		for t in triples:
			edge_labels[t[0] + "|||" + t[2]] = t[1]
		return edge_labels 

	G_pred_labels = get_edge_labels(predicted_triples)
	G_truth_labels = get_edge_labels(ground_truth_triples)

	def get_graph_embs(graph, labels):
		embs = []
		#paths = get_all_paths(graph, labels)
		for x in range(N_RANDOM_WALKS):
			p = random_walk(graph, labels)
			#if p is None:
			#	emb = np.zeros((768)).astype(float)
			#else:
			emb = bc.encode([' '.join(p)])[0]
			embs.append(emb)
		
		return np.asarray(embs)

	cosine_score = 0
	#embs_g = np.mean(get_graph_embs(G_truth, G_truth_labels), axis=0)
	#embs_g = get_graph_embs(G_truth, G_truth_labels)
	#embs_p = get_graph_embs(G_pred, G_pred_labels)
	#if embs_p is not None:

	

	if len(ground_truth_triples) > 0:
		if len(predicted_triples) == 0:
			return 0

		embs_p = [bc.encode([' '.join(p)])[0] for p in predicted_triples]
		embs_g = [bc.encode([' '.join(g)])[0] for g in ground_truth_triples]


		similarity_matrix = np.zeros([len(embs_g), len(embs_p)])
		
		cosine_scores = []
		seen_ps = set()
		for gi, g in enumerate(embs_g):

			for pi, p in enumerate(embs_p):
				c = 1 - cosine(p, g)
				similarity_matrix[gi, pi] = c

		#np.set_printoptions(precision=3)


		for i in range(len(embs_g)):

			max_val = np.amax(similarity_matrix)

			cosine_scores.append(max_val)
			

			row, col = np.unravel_index(np.argmax(similarity_matrix, axis=None), similarity_matrix.shape)


			similarity_matrix[row, :] = 0.0
			similarity_matrix[:, col] = 0.0


			# Find the closest non-visited embedding in embs_p
			#best = 0.0
			#for pi, p in enumerate(embs_p):
				#if pi in seen_ps:
				#	continue
				#c = 1 - cosine(p, g)
				#if c > best:
				#	best = c
				#	seen_ps.add(i)
			#cosine_scores.append(best)



		cosine_score = sum(cosine_scores) / len(cosine_scores)
		precision = max(1.0, 1.0 * len(embs_g) / len(embs_p))
		triples_score = precision * cosine_score


		if False:

			#cosine_score = 1 - cosine(embs_p, embs_g)
			cosine_scores = []

			#for x in range(N_RANDOM_WALKS):
			embs_p = get_graph_embs(G_pred, G_pred_labels)			
			embs_g = get_graph_embs(G_truth, G_truth_labels)	

			for p, g in zip(embs_p, embs_g):
				#for p in embs_p:		
				c = 1 - cosine(p, g)

				#c = np.sum(p * g) / np.linalg.norm(p * g)
				#print(c)

				if np.isnan(c):
					c = 0 # i.e. no predicted triples for this doc
					
				cosine_scores.append(c)

			cosine_score = sum(cosine_scores) / len(cosine_scores)


		# for g in embs_g:
		# 	max_c = 0
		# 	for p in embs_p:
		# 		c = 1 - cosine(p, g)
		# 		#print(c, max_c)
		# 		if c > max_c:
		# 			max_c = c 			
		# 	cosine_scores.append(max_c)

		#for p, g in zip(embs_p, embs_g):
		#	cosine_scores.append(1 - cosine(p, g))
		#cosine_score = sum(cosine_scores) / len(cosine_scores)

		#cosine_score = sum(cosine_scores) / len(cosine_scores)

	#print(embs_p)

	#print(embs_p)
	#print(embs_g)

	# (e1 + e2) / 2
	# return cosine_score
	return triples_score


	#exit()
	#return sum(costs) / len(costs)
	#return ged.similarity(result)

def random_walk(graph, edge_labels):
	# https://stackoverflow.com/questions/22150247/more-efficient-way-of-running-a-random-traversal-of-a-directed-graph-with-networ
	# Get a random path from the graph
	all_paths = dict(nx.all_pairs_shortest_path(graph))

	# short_paths = []

	# for k in all_paths:
	# 	for key, val in k[1].items():
	# 		short_paths.append(val)


	source, target = None, None
	if len(edge_labels) == 0:
		return None

	
	r = []
	n = 0
	while len(r) == 0:
		source = random.choice(list(all_paths.keys()))
		r = list(item for item in all_paths[source].keys() if item != source)
		n += 1
		if n > 20:
			return None
	target = random.choice(r)
		

	# Random path is at
	random_path = all_paths[source][target]

	final_paths = []

	# final_paths = []
	# for p in short_paths:
	# 	path = []
	# 	if len(p) == 1: continue
	# 	for i in range(len(p)):
	# 		path.append(p[i])
	# 		if i == len(p)-1: continue		
	# 		path.append(edge_labels[p[i] + "|||" + p[i+1]])
	# 	final_paths.append(path)

	final_path = []
	for i in range(len(random_path)):
		final_path.append(random_path[i])
		if i == len(random_path)-1: continue		
		final_path.append(edge_labels[random_path[i] + "|||" + random_path[i+1]])

	return  final_path


def main():
	for industry in industries:
		print("\nIndustry: %s" % industry)

		with open('../../datasets/%s/test.csv' % industry, 'r') as f:
			num_docs = len(f.readlines())

		with open('../../datasets/%s/ground_truth_triples_test.csv' % industry, 'r') as f:
			reader = csv.reader(f)
			ground_truth_triples = parse_file(reader, num_docs)
			print("===")
			
		for model in models:
			with open('../%s/output/%s.csv' % (model, industry), 'r') as f:
				reader = csv.reader(f)
				predicted_triples = parse_file(reader, num_docs)

			scores = []
			for i, (p, g) in enumerate(zip(predicted_triples, ground_truth_triples)):
				score = evaluate_triples(p, g)
				scores.append(score)
				sys.stdout.write("\r  Model: %s\t%d / %d" % (model, i, len(predicted_triples)))
				sys.stdout.flush()

			score = sum(scores) / len(scores)
			print("\r  Model: %s                                   " % model)
			print("\r    Score : %.4f" % score)
			print()

if __name__ == "__main__":
	main()