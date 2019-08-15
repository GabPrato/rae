import h5py
import numpy as np

###### Parameters #####
TEXT_FILE = 'bookcorpus.txt' # One sentence per line, sentences must already be tokenized.
MAX_SENTENCE_LENGTH = 100
DATASET_FILE = 'bookcorpus.hdf5'
VOCABULARY_FILE = 'vocabulary.txt'
TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.1 # test ratio is then determined by leftovers, in this case would be 0.1


###### Load text file #####
voc = {}
sentences = {}
for i in range(2, MAX_SENTENCE_LENGTH + 1):
    sentences[i] = []

with open(TEXT_FILE, 'r') as f:
    for line in f:
        sent = line.rstrip().split()
        len_sent = len(sent)
        if len_sent >= 2 and len_sent <= MAX_SENTENCE_LENGTH:
            sentences[len_sent].append(sent)
            for word in sent:
                voc[word] = voc.get(word, 0) + 1
        
print(f'Found {len(voc)} unique tokens.')


###### Load GloVe embeddings #####
glove_embeddings = {}
with open('glove.42B.300d.txt') as f:
    for line in f:
        word, vec = line.split(' ', 1)
        if word in voc:
            glove_embeddings[word] = np.array(list(map(float, vec.split())))
            
are_in = 0
not_in = 0
for k, v in voc.items():
    if k in glove_embeddings:
        are_in += v
    else:
        not_in += v
print(f'{(are_in * 100) / (are_in + not_in)}% of the tokens have an embedding')
print(f"{(not_in * 100) / (are_in + not_in)}% of the tokens don't")


###### Set token ids #####
ids = {}
count = 1
for k in glove_embeddings.keys():
    if k not in ids:
        ids[k] = count
        count += 1
        
print(f'{len(ids)} token ids (vocabulary size).')


###### Create embeddings dataset #####
dataset = h5py.File(DATASET_FILE,'w')
emb_dataset = np.zeros((len(ids) + 1, 300), dtype='float32')
for k, v in ids.items():
    emb_dataset[v] = np.array(glove_embeddings[k], dtype='float32')
dataset.create_dataset("embeddings", data=emb_dataset)


###### Save vocabulary #####
voc_pair = []
for k, v in ids.items():
    voc_pair.append((v, k))
voc_pair.sort()

with open(VOCABULARY_FILE, 'w') as f:
    f.write('<UNKNOWN> ')
    for _id, word in voc_pair:
        f.write(f'{word} ')


###### Save sequences to dataset #####
for k, v in sentences.items():
    ided_sentences = np.zeros((len(v), k), dtype=np.uint32)
    for i, sent in enumerate(v):
        for j, word in enumerate(sent):
            ided_sentences[i, j] = ids.get(word, 0)      
    
    np.random.shuffle(ided_sentences)
    
    train_size = int(ided_sentences.shape[0] * TRAIN_RATIO)
    validation_size = int(ided_sentences.shape[0] * VALIDATION_RATIO)
    
    if train_size > 0:
        dataset.create_dataset(f'train/{k}', data=ided_sentences[:train_size])
    if validation_size > 0:
        dataset.create_dataset(f'validation/{k}', data=ided_sentences[train_size : train_size + validation_size])
    if (train_size + validation_size) < ided_sentences.shape[0]:
        dataset.create_dataset(f'test/{k}', data=ided_sentences[train_size + validation_size:])
        
dataset.close()

