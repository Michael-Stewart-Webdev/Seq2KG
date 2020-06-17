import nltk
from nltk.tag import pos_tag, map_tag
np_grammar = 	r"""
					NP: 
						{<NOUN>+}
						{<ADJ>+<NOUN>+}
												
					RP: 
						{<VERB|ADP>+}
				"""
chunk_parser = nltk.RegexpParser(np_grammar)

def get_phrase_type(phrase):
	tagged_phrase = nltk.pos_tag(phrase)
	tagged_phrase = [(word, map_tag('en-ptb', 'universal', tag)) for word, tag in tagged_phrase]
	result = chunk_parser.parse(tagged_phrase)
	phrase_type = str(result[0])[1:3]
	return phrase_type

def get_all_permutations(tokens, doc_idx, max_size=3, max_dist=7):
	head_sizes = [x for x in range(1, max_size + 1)]
	rel_sizes = [x for x in range(1, max_size + 1)]
	tail_sizes = [x for x in range(1, max_size + 1)]
	triples = []
	total = 0
	

	for h in head_sizes:
		
		for r in rel_sizes:
			for t in tail_sizes:				
				for head_idx in range(0, len(tokens) - r - t):
					#print("head: %s:%s" %  (head_idx, h))

					phrase_type = get_phrase_type(tokens[head_idx:head_idx + h])
					if phrase_type != "NP":
						continue

					for rel_idx in range(head_idx + h, min(head_idx + h + max_dist, len(tokens) - r)):
						phrase_type = get_phrase_type(tokens[rel_idx:rel_idx + r])
						if phrase_type != "RP":
							continue

						for tail_idx in range(head_idx + h + r, min(head_idx + h + r + max_dist,len(tokens))):
							if(tail_idx + t > len(tokens)):
								continue
							#print("tail: %s:%s" %  (tail_idx, t))

							phrase_type = get_phrase_type(tokens[tail_idx:tail_idx + t])
							if phrase_type == "NP":														
								total += 1
					#print((tokens[head_idx:head_idx + h], tokens[rel_idx:rel_idx + r], tokens[tail_idx: tail_idx + t]))
								triples.append([doc_idx,
												" ".join(tokens[head_idx:head_idx + h]), 
												" ".join(tokens[rel_idx:rel_idx + r]),
												" ".join(tokens[tail_idx: tail_idx + t])])
								#print("\r%d" % (total), end="")

	#print()
	#print(len(triples))
	return triples
if __name__ == "__main__":
	get_all_permutations(["Barrack", "spoke", "to", "Michelle", "at", "the", "Whitehouse"])