import sys
import os
sys.path.append('../')
import config
from tqdm import tqdm


def convertAnnData(annotate):
    ann_data = []
    for line in annotate[:-1]:
        entity = line.split('\t')[-1]
        start = int(line.split('\t')[1].split(' ')[1])
        end = int(line.split('\t')[1].split(' ')[-1])
        ann_data.append({entity: (start, end)})
    return ann_data


def train():
    texts = []
    anns = []
    file_ids = [f[:-4] for f in os.listdir(config.train_data_root) if f[-4:] == '.txt']
    for file_id in tqdm(file_ids):
        with open(config.train_data_root + '{}.txt'.format(file_id)) as f:
            texts.append(f.read())
        with open(config.train_data_root + '{}.ann'.format(file_id)) as f:
            anns.append(convertAnnData(f.read().split('\n')))
    assert len(texts) == len(anns), 'must have the same length texts and anns'
    return texts, anns


def valid():
    texts = []
    anns = []
    file_ids = [f[:-4] for f in os.listdir(config.valid_data_root) if f[-4:] == '.txt']
    for file_id in tqdm(file_ids):
        with open(config.valid_data_root + '{}.txt'.format(file_id)) as f:
            texts.append(f.read())
        with open(config.valid_data_root + '{}.ann'.format(file_id)) as f:
            anns.append(convertAnnData(f.read().split('\n')))
    assert len(texts) == len(anns), 'must have the same length texts and anns'
    return texts, anns


def test():
    texts = []
    anns = []
    file_ids = [f[:-4] for f in os.listdir(config.test_data_root) if f[-4:] == '.txt']
    for file_id in tqdm(file_ids):
        with open(config.test_data_root + '{}.txt'.format(file_id)) as f:
            texts.append(f.read())
        with open(config.test_data_root + '{}.ann'.format(file_id)) as f:
            anns.append(convertAnnData(f.read().split('\n')))
    assert len(texts) == len(anns), 'must have the same length texts and anns'
    return texts, anns
