# build_triples.py
# Converts a string into a set of triples.

from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

from nltk.tag import pos_tag, map_tag
# np_grammar = 	r"""
# 					NP: 
# 						{<NOUN>+}
# 						{<ADJ>+<NOUN>+}
												
# 					RP: 
# 						{<VERB|ADP>+}
# 				"""

np_grammar = r"""
					NP: 
						{<CD>*<NN|NNS|NNP|NNPS>+<NN|NNS|NNP|NNPS|POS>*}
						
					RP: 
						{<VB|VBD|VBG|VBN|VBP|IN>+}
				"""

# Retrieve all permutations of noun and relation phrases, such that triples consist of:
# NP -> RP -> NP
# 
# phrases: A list of phrases, whereby each phrase is (phrase: string, phrase_type: string)
# max_dist: Maximum distance between head and tail (in tokens). Must be greater than 0.
# max_rp_dist: Maximum number of tail phrases per head phrase. May be -1 to signify no maximum.
def get_permutations(phrases, max_dist=15, max_rp_dist=-1):
	triples = []
	if len(phrases) == 0:
		return []

	# Retrieve the index of the next head term by jumping to the next NP, starting from start_idx.
	def get_next_head_idx(start_idx, phrases):
		for i, (phrase_1, phrase_type_1) in enumerate(phrases[start_idx:]):
			if i == (len(phrases[start_idx:]) - 1):
				return -1
			if phrase_type_1 == "NP":
				return start_idx + i

	start_idx = get_next_head_idx(0, phrases)
	seen_rels = [set()] * len(phrases)
	seen_tails = [set()] * len(phrases)

	

	while True:
		if start_idx == -1:
			return triples

		phrase_1, phrase_type_1 = phrases[start_idx]		
		current_head = phrase_1

		current_rel  = None 
		if len(seen_rels[start_idx]) == 0:
			seen_rels[start_idx] = set()
		if len(seen_tails[start_idx]) == 0:
			seen_tails[start_idx] = set()

		end_idx = min(len(phrases), start_idx + max_dist)
		for idx, (phrase_2, phrase_type_2) in enumerate(phrases[start_idx + 1 : end_idx ]):	
			
			if phrase_type_2 == "NP" and current_rel is not None:
				triples.append([current_head, current_rel, phrase_2])
				seen_tails[start_idx].add(phrase_2)
				if max_rp_dist > 0 and len(seen_tails[start_idx]) >= max_rp_dist:
					start_idx = get_next_head_idx(start_idx + 1, phrases)			
					break
						
			elif phrase_type_2 == "RP" and current_rel is None and phrase_2 not in seen_rels[start_idx]:
				current_rel = phrase_2 
				seen_rels[start_idx].add(phrase_2)

			if (idx) == (len(phrases[start_idx + 1: end_idx]) - 1) and current_rel is None:
				start_idx = get_next_head_idx(start_idx + 1, phrases)				
				break	

	#for t in triples:
	#	print t 
	#print "%d triples total." % len(triples)
	return triples

# Evaluate the model on a list of strings and return the list of lists of triples predicted by the model.
def get_coreffed_docs_from_doc(doc):
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


	return ' '.join(spacy_tokens)



# Build triples from a list of documents.
# The documents should ideally be sentences with coref resolution applied for maximum performance.
def build_triples(documents, add_doc_idx=True, verbose=True):
	'''documents = []

	with codecs.open('documents.txt', 'r', 'utf-8') as f:
		documents = [line.strip() for line in f.readlines()]'''
	#df = pd.read_csv('catering_only.csv', encoding = 'utf8')
	#for row in df[df.columns[1]]:
	#	documents.append(row)

	all_triples = []
	all_sents   = []
	for doc_idx, doc in enumerate(documents):

		triples_this_doc = 0
		#doc = ''.join([i if ord(i) < 128 else ' ' for i in doc])

		sents = sent_tokenize(doc)

		# Scan each sentence within each entry
		for sent_index, sent in enumerate(sents):

			all_sents.append(sent)

			noun_phrases 				= []
			relation_phrases 			= []
			phrases = []

			tokens = word_tokenize(sent)
			#tokens = word_tokenize(sent)
			#print(tokens)
			tagged_sent = nltk.pos_tag(tokens)
			#tagged_sent = [(word, map_tag('en-ptb', 'universal', tag)) for word, tag in tagged_sent]

			chunk_parser = nltk.RegexpParser(np_grammar)
			result = chunk_parser.parse(tagged_sent)
			#print(tagged_sent)


			
			for chunk in result:
				phrase_type = str(chunk)[1:3]

				# Noun phrases appear as trees in the result
				if chunk.__class__.__name__ == "Tree":


					# Get the entire phrase in the chunk. process_words applies lemmatization and stop words if toggled on.
					#phrase = [w[0] for w in chunk if all([c.isalpha() for c in w])]
					phrase = [w[0] for w in chunk]
					#if phrase_type == "NP":
					#	noun_phrases.append(phrase)
					#elif phrase_type == "RP":
					#	relation_phrases.append(phrase)
					if len(phrase) > 0 and len([x for x in phrase if x.isalpha() or x.isnumeric()]) > 0:
						phrases.append((" ".join(phrase), phrase_type))

			permutations = get_permutations(phrases)
			for p in permutations:
				#if triples_this_doc <= 10:
				if add_doc_idx:
					all_triples.append([doc_idx] + p)
				else:
					all_triples.append(p)
				triples_this_doc += 1

		#print all_sents
		if verbose:
			print ("\r%d / %d complete." % ((doc_idx + 1), len(documents)), end="" )
	if verbose:
		print()

	if add_doc_idx == False: # Ensure triples are unique (they might not be due to coref resolution)
		unique_triples = set()
		final_triples = []
		for t in all_triples:
			s = ' '.join(t[0]) + ' '.join(t[1]) + ' '.join(t[2])
			if s not in unique_triples:
				unique_triples.add(s)
				final_triples.append(t)
		return final_triples
	return all_triples
	# df = pd.DataFrame(data=all_triples)#, columns=["sent_index", "s1", "r", "s2", "s1_id", "s2_id"])
	# df.to_csv('triples.csv', header=False)

	# df = pd.DataFrame(data=all_sents)#, columns=["sent_index", "s1", "r", "s2", "s1_id", "s2_id"])
	# df.to_csv('sents.csv', header=False)



# How to import to Neo4j:
'''
LOAD CSV FROM "file:///triples.csv" as line

MERGE (sub:Subject {name:line[2]})
MERGE (obj:Subject {name:line[4]})
WITH sub, obj, line
CALL apoc.merge.relationship(sub,line[3],{},{},obj) YIELD rel
RETURN COUNT(*);
'''


# To delete:
'''
MATCH (n)
DETACH DELETE n
'''

if __name__ == "__main__":
	main()