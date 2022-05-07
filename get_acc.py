import numpy as np
import pickle
import json

from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')

with open('data-0505/res4.pkl', 'rb') as fin:
    res, test_encodings = pickle.load(fin)

starts = np.argmax(res[0][0], axis = -1)
ends = np.argmax(res[0][1], axis = -1)

print(starts[:100])
print(ends[:100])
print(res[1][0][:100])
print(res[1][1][:100])
print(len(starts))
num = 0

for ps, pe, start, end in zip(starts, ends, res[1][0], res[1][1]):
    if ps == start and pe == end:
        num += 1

print(num)

with open('data-0505/test.pkl', 'rb') as fin:
    test = pickle.load(fin)

preds = []

ind = 0
for line in test:
    cr = {}
    cr['content'] = line[1]
    labels = json.loads(line[-1])
    annotations = {}
    for label in labels:
        if not label['type'] in annotations:
            annotations[label['type']] = tokenizer.convert_ids_to_tokens(test_encodings['input_ids'][ind][starts[ind]: ends[ind]+ 1])
            tr = ''
            for i, t in enumerate(annotations[label['type']]):
                #if not t.startswith('##'):
                if t.startswith('Ġ'):
                    if i != 0:
                        tr += ' '
                    tr += t
                else:
                    tr += t
            tr = tr.replace('Ġ', '')
            annotations[label['type']] = [tr]
        ind += 1
    cr['annotations'] = annotations
    cr['golden'] = labels
    preds.append(cr)

with open('data-0505/preds4.pkl', 'wb') as fout:
    pickle.dump(preds, fout)
        


