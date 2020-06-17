lines = []
with open('output.json', 'r') as f:
	lines = [l.strip() for l in f]


train = lines[:int(0.8 * len(lines))]
dev   = lines[int(0.8 * len(lines)):int(0.9 * len(lines))]
test  = lines[int(0.9 * len(lines)): ]

with open('train.json', 'w') as f:
	f.write("\n".join(train))
with open('dev.json', 'w') as f:
	f.write("\n".join(dev))
with open('test.json', 'w') as f:
	f.write("\n".join(test))