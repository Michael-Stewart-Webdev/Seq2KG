import spacy
import neuralcoref

# Splits annotated Redcoat documents into annotated sentences.

# Parse the document-level annotations and convert them into sentence-level annotations.
# Perform coference resolution so that the entities in previous sentences carry through to later
# sentences.
# Fix up the tag numbers, i.e. if head_3 is the first tag in sentence 2, it becomes head_1 and so on.
def redcoat_to_sents(redcoat_data):
	print("Converting document-level annotations into sentence-level annotations...", end="")
	data = []
	
	nlp = spacy.load('en')
	coref = neuralcoref.NeuralCoref(nlp.vocab)
	nlp.add_pipe(coref, name='neuralcoref')

	final_training_data = []
	non_annotated_docs = 0

	for doc_idx, obj in enumerate(redcoat_data):
	
		orig_tokens = obj['tokens']
		orig_mentions = obj['mentions']
		if len(orig_mentions) == 0:
			non_annotated_docs += 1
			continue

		doc = nlp(" ".join(obj['tokens']))

		spacy_tokens = [str(w) for w in doc]

		# Build a mapping between the original tokens and the Spacy tokens, which are
		# tokenized differently.
		orig_to_spacy = {}
		
		i = 0
		offset = 0
		for i in range(len(orig_tokens)):			
			spacy_token = str(doc[i + offset])
			orig_token = orig_tokens[i]

			if not orig_token.startswith(spacy_token):				
				while not orig_token.startswith(str(doc[i + offset])):	
					offset += 1
					#print(orig_token, str(doc[i]), str(doc[i+offset]))				
					
			orig_to_spacy[i] = i + offset			
			spacy_token = str(doc[i + offset])		
			#print(i, orig_token, spacy_token, offset, orig_to_spacy[i])
		orig_to_spacy[i+1] = i + 1 + offset # Add the last token



		# Construct a new document object using the spacy-tokenized data with updated label positions and coref.

		tagged_doc = []
		for token in spacy_tokens:
			tagged_doc.append([token, []])

		# Convert the format to a more usable data structure:
		# [word, labels]
		for mention in orig_mentions:
			start = mention['start']
			end = mention['end']
			labels = [l.split("/")[1] for l in mention['labels'] if "_" in l]	
			
			spacy_start = orig_to_spacy[start]
			spacy_end = orig_to_spacy[end]
			for i in range(spacy_start, spacy_end):
				if not tagged_doc[i][0] == ".": # Remove labels from full stops, which are joined to words in Redcoat's tokenizer
					tagged_doc[i][1] = labels			

		# Perform neural coreference resolution, copying the labels of the main cluster to the
		# tokens that refer back to it.
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

					labels = tagged_doc[m.start + offset][1]
					if(len(labels) == 0 and len(tagged_doc) > m._.coref_cluster.main.start): # Only swap the labels if this word does not already have labels
						labels = tagged_doc[m._.coref_cluster.main.start][1]

					main_coref_list = [[w, labels] for w in main_coref]

					tagged_doc = tagged_doc[:m.start + offset] + main_coref_list + tagged_doc[m.end + offset:]
					offset += new_len - orig_len
					#print(m, main_coref_list)
		
		# Split up the object into sentences.
		sents = []
		current_sent = []
		for word, labels in tagged_doc:
			if word == "." and len(current_sent) > 0:
				sents.append(current_sent)
				current_sent = []
				continue
			current_sent.append([word, labels])
		if len(current_sent) > 0:
			sents.append(current_sent)
	

		# Fix the label indexes, i.e. head_3 in the second sentence should be
		# tag_1, and so on.
		aligned_sents = []
		for s in sents:
			label_map = {}
			labels_seen = [{"head": 0, "rel": 0, "tail": 0} for x in range(10)]
			for word, labels in s:
				for label in labels:
					label_type, idx = label.split("_")
					labels_seen[int(idx) - 1][label_type] += 1

			# Build a list of label indexes that are present as a head, rel, and tail in the sentence.
			complete_labels = []
			for k, v in enumerate(labels_seen):
				if v["head"] > 0 and v["rel"] > 0 and v["tail"] > 0:
					complete_labels.append(k)

			aligned_sents.append([])
			for word, labels in s:
				new_labels = []
				for label in labels:
					label_type, idx = label.split("_")
					idx = int(idx) - 1
					if idx in complete_labels:
						new_idx = complete_labels.index(idx) + 1

						# new_labels.append(label_type)
						# for lx in range(1, new_idx + 1):
						# 	new_labels.append(label_type + ''.join([str(p) + '/' for p in range(1, lx)]) + "/" + str(lx))
						
						new_labels.append(label_type)
						new_labels.append(label_type + "/" + str(new_idx))


				aligned_sents[-1].append([word, new_labels])

		# for s in sents:
		# 	for w in s:
		# 		print(w[0], w[1])
		# print("======")
		# for s in aligned_sents:
		# 	for w in s:
		# 		print(w[0], w[1])
		# exit()
		

		# Convert the sentences into the mention-level typing format.

		def tagged_sents_to_mentions(tagged_sents):
			mentions_data = []
			for s in tagged_sents:
				tokens = [w[0] for w in s]
				mentions = []
				current_labels = set()
				current_start = -1
				labels_seen = set()

				for i, (word, labels) in enumerate(s):

					labels = [l for l in labels if l not in labels_seen]
					
					#


					if len(current_labels) == 0:
						if len(labels) > 0:
							current_labels = labels
							current_start = i
					elif set(labels) != set(current_labels):

						mentions.append({'start': current_start, 'end': i, 'labels': current_labels})
						for l in current_labels:
							labels_seen.add(l)

						current_labels = labels
						current_start = i

					# Handle the last token correctly
					if i == (len(s) - 1) and len(current_labels) > 0:
						mentions.append({'start': current_start, 'end': i+1, 'labels': current_labels})
						break
						


				mentions_data.append({'tokens': tokens, 'mentions': mentions })
			
			return mentions_data
					

		training_data = tagged_sents_to_mentions(aligned_sents)
		
		for doc in training_data:
			#print(":____")
			#print(doc)
			if len(doc['mentions']) > 0:
				final_training_data.append(doc)
			#for m in doc['mentions']:
			#	print(doc['tokens'][m['start']:m['end']], m['labels'])
		
		print("\rParsing annotations... %s / %s (%d not annotated)" % (doc_idx, len(redcoat_data), non_annotated_docs), end="")
	print()
		
	return final_training_data