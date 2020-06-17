import jsonlines, sys

# Return a list of triples that are present in a Redcoat document.
def get_redcoat_triples(doc):
	mentions = doc['mentions']
	triples = [[None, None, None] for x in range(10)]
	ordering = {"head": 0, "rel": 1, "tail": 2}
	for m in mentions['triples']:
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


# Retrieve the list of ground truth triples for a given dev set.
def get_ground_truth_triples(dev_set):
	ground_truth_triples = []
	for doc_idx, doc in enumerate(dev_set):
		ground_truth_triples += [[doc_idx] + gt + [1] for gt in get_redcoat_triples(doc)]
	return ground_truth_triples




with jsonlines.open('triples/train.json') as reader:
	train_triples = { line['doc_idx']: line for line in reader if len(line['mentions']) > 0}



with jsonlines.open('triples/dev-test.json') as reader:
	lines = [line for line in reader]
	dev_triples = { int((line['doc_idx']/2)): line for line in lines if line['doc_idx'] % 2 == 0 and len(line['mentions']) > 0}
	#dev_triples = [ line for line in lines if line['doc_idx'] % 2 == 0 ]
	test_triples = { int((line['doc_idx'] - 1)/2): line for line in lines if line['doc_idx'] % 2 != 0 and len(line['mentions']) > 0}
	#test_triples = [ line for line in lines if line['doc_idx'] % 2 != 0 ]







with jsonlines.open('entity_typing/train.json') as reader:
	train_et = [line for line in reader]

with jsonlines.open('entity_typing/dev.json') as reader:
	dev_et = [line for line in reader]

with jsonlines.open('entity_typing/test.json') as reader:
	test_et = [line for line in reader]



# Join

def merge_dataset(dataset_triples, dataset_et):
	merged_data = []
	bad = 0

	for doc_idx in dataset_triples:


		tokens_triples   = dataset_triples[doc_idx]['tokens']
		mentions_triples = dataset_triples[doc_idx]['mentions']
		tokens_et        = dataset_et[doc_idx]['tokens']
		mentions_et      = dataset_et[doc_idx]['mentions']




		
		# Fix differences in tokenisation between redcoat/bbn original dataset
		if tokens_triples != tokens_et:
			#print(" ".join(tokens_triples))
			#print(" ".join(tokens_et))
			#print('---')

			#print(mentions_triples)
			diff = 0
			for i, token in enumerate(tokens_et):
				#print(i, token, tokens_triples[i+diff])
				if tokens_et[i] != tokens_triples[i+diff]:
					#print(tokens_et[i], tokens_triples[i+diff])
					diff += 1
					#print(diff)
					for j, m in enumerate(mentions_triples):
						if mentions_triples[j]['start'] > (i):
							#print(m['start'], i+diff)

							mentions_triples[j]['start'] = mentions_triples[j]['start'] -1
						if mentions_triples[j]['end'] > (i):
							mentions_triples[j]['end'] = mentions_triples[j]['end'] -1
						#print(tokens_et[mentions_triples[j]['start']:mentions_triples[j]['end']])


			#print(mentions_triples)

			#bad += 1
			#continue
		#assert tokens_triples == tokens_et 

		tokens = tokens_et

		# Add "MISC" entity type to the triple heads/tails if they are not already listed as an entity
		#for i, token in enumerate(tokens):

		new_mentions_et = mentions_et[:]

		for j, m in enumerate(mentions_triples):


			start = m['start']
			end   = m['end']
			print(m)
			if not (any('/head' in label for label in m['labels']) or any('/tail' in label for label in m['labels'])):
				continue

			new_start, new_end = start, end

			in_et = False
			for k, n in enumerate(mentions_et):
				start2 = n['start']
				end2 = n['end']

				if start >= start2:
					if end <= end2:
						in_et = True

					elif end > end2 and end2 > start:
						new_start = end2
						new_end = end
				if start2 > start:
					if end2 > end and end > start2:
						new_start = start 
						new_end = start2

			
			if not in_et:
				new_mentions_et.append({'start': new_start, 'end': new_end, 'labels': ['/MISC']})



		
		# Map labels in triples
		for k, m in enumerate(mentions_triples):
			updated_labels = []
			for l in m['labels']:
				
				if '/' not in l:
					continue

				(triple_num, triple_part) = l.split('/')

				num = triple_num[1:]

				if triple_part not in updated_labels:
					updated_labels.append(triple_part)
				l2 = triple_part + "/" + num


				updated_labels.append(l2)
			mentions_triples[k]['labels'] = updated_labels






		# Print the tokens/labels
		verbose = False
		if verbose:

			labelled_tokens = [ [token, [], []] for token in tokens]

			for j, m in enumerate(new_mentions_et):
				start = m['start']
				end = m['end']
				for x in range(start, end):
					labelled_tokens[x][1] = m['labels']

			for j, m in enumerate(mentions_triples):
				start = m['start']
				end = m['end']
				for x in range(start, end):
					labelled_tokens[x][2] = m['labels']

			for t in labelled_tokens:
				if len(t[1]) > 0 or len(t[2]) > 0:
					print("%s %s %s" % (t[0].ljust(15)[:15], " ".join(t[1])[:22].ljust(25), " ".join(t[2])[:25].ljust(25)))
			print("-" * 30)




		merged_data.append({'tokens': tokens, 'mentions': {'triples': mentions_triples, 'entity_types': new_mentions_et } })
	#print(bad, len(dataset_triples))
	return merged_data

merged_data_train = merge_dataset(train_triples, train_et)
merged_data_dev   = merge_dataset(dev_triples, dev_et)
merged_data_test  = merge_dataset(test_triples, test_et)

with jsonlines.open('../train.json', 'w') as writer:
	writer.write_all(merged_data_train)

with jsonlines.open('../dev.json', 'w') as writer:
	writer.write_all(merged_data_dev)

with jsonlines.open('../test.json', 'w') as writer:
	writer.write_all(merged_data_test)

#ground_truth_triples = get_ground_truth_triples(merged_data_dev)

# with open('../ground_truth_triples.csv', 'w') as f:
# 	f.write("index,s1,r,s2\n")
# 	for t in ground_truth_triples:
# 		f.write('"%d","%s","%s","%s"' % (t[0], t[1], t[2], t[3]))
# 		f.write("\n")