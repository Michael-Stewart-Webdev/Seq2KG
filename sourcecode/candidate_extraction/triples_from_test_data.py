from triples_from_text import extract_triples

import csv

industries = ['bbn']#, 'automotiveEngineering']

def main():

	for industry in industries:
		print("Evaluating model on %s dataset..." % industry)
		test_data_filename = '../../datasets/%s/test.csv' % industry

		data = []

		# Read the contest data into separate objects
		with open(test_data_filename, 'r') as f:
			reader = csv.reader(f)
			csv_rows = list(reader)
		for row in csv_rows: # skip header
			idx = int(row[0])
			content = row[1]
			data.append(content)

	

		with open('output/%s.csv' % industry, 'w') as f:
			writer = csv.writer(f)			
		
			triples = []
			for d in data:
				ts = extract_triples(d)
				print(ts)
				triples.append(ts)

			writer.writerow(['index','s1','r','s2'])
			for doc_idx, doc in enumerate(triples):
				for t in doc:					
					writer.writerow([doc_idx] + [t[0]] + [t[1]] + [t[2]])

if __name__=="__main__":
	main()