from build_triples import build_triples

import csv
contest_data_filename = '../../datasets/icdm_contest_data.csv'

import spacy, neuralcoref

nlp = spacy.load('en')
coref = neuralcoref.NeuralCoref(nlp.vocab)
nlp.add_pipe(coref, name='neuralcoref')


def coref_substitute(text):
	doc = nlp(text)
	spacy_tokens = [str(w) for w in doc]

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

	return " ".join(spacy_tokens)


def main():

	data = {}

	# Read the contest data into separate objects
	with open(contest_data_filename, 'r') as f:
		reader = csv.reader(f)
		csv_rows = list(reader)
	print("Loading data and peforming coref resolution...")
	for row in csv_rows[1:]: # skip header
		idx = int(row[0])
		content = row[1]
		industry = row[2]
		if industry not in data:
			data[industry] = []		

		#content = coref_substitute(content)	
	
		data[industry].append(content)
		assert len(data[industry]) == (idx + 1)
		print("\r%s" % idx, end="")


	with open('output/submission.csv', 'w') as f:
		writer = csv.writer(f)
		writer.writerow(['industry','index','s1','r','s2'])	
		for industry in data:		
			triples = build_triples(data[industry])

					
			for t in triples:					
				writer.writerow([industry] + [t[0]] + [t[1]] + [t[2]] + [t[3]])

if __name__=="__main__":
	main()