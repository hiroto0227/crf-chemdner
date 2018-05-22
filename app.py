from flask import Flask
import os
from tqdm import tqdm
from scripts.models import CRFModel
from scripts.featurize import convertSentenceToFeatures, convertAnnToLabels, convertAnnData, convertLabelToAnnData
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import collections

app = Flask(__name__)
crf_model = CRFModel()


@app.route('/')
def index():
    return ('/init, /evaluate, /fit, /predict')


@app.route('/init')
def init():
    data_file_path = './datas/chemdner-corpora/chemdner_standoff/'
    for mode in ['train', 'test', 'devel']:
        # init
        features = []
        labels = []
        file_ids = [f[:-4] for f in os.listdir(data_file_path + mode) if f[-4:] == '.txt']
        for file_id in tqdm(file_ids):
            with open(data_file_path + mode + '/{}.txt'.format(file_id)) as f:
                sentence = f.read()
            with open(data_file_path + mode + '/{}.ann'.format(file_id)) as f:
                ann_data = convertAnnData(f.read().split('\n'))
            features.extend(convertSentenceToFeatures(sentence))
            labels.extend(convertAnnToLabels(ann_data, len(sentence)))
        with open('./datas/x_{}.pickle'.format(mode), 'wb') as f:
            pickle.dump(features, f)
        with open('./datas/y_{}.pickle'.format(mode), 'wb') as f:
            pickle.dump(labels, f)
    if os.path.exists('./models/conll2002-esp.crfsuite'):
        crf_model.crf
    return '200'


@app.route('/fit')
def fit():
    # train_data read
    print('---------- load data ----------------')
    with open('./datas/x_train.pickle', 'rb') as f:
        features = pickle.load(f)
    with open('./datas/y_train.pickle', 'rb') as f:
        labels = pickle.load(f)
    with open('./datas/x_devel.pickle', 'rb') as f:
        features.extend(pickle.load(f))
    with open('./datas/y_devel.pickle', 'rb') as f:
        labels.extend(pickle.load(f))
    # fit model
    crf_model.fit(features, labels)
    return 'trainning end'


@app.route('/predict')
def predict():

    return 200


@app.route('/evaluate')
def evaluate():
    with open('./datas/x_test.pickle', 'rb') as f:
        test_features = pickle.load(f)
    pred_labels = crf_model.predict(test_features)
    with open('./datas/y_test.pickle', 'rb') as f:
        labels = pickle.load(f)
    sentence = ''.join([d['1g'] for d in test_features])
    # true label
    entities = []
    for ann in convertLabelToAnnData(sentence, labels):
        for k, v in ann.items():
            entities.append(k)
    counter = collections.Counter(entities)
    # pred label
    pred_entities = []
    for ann in convertLabelToAnnData(sentence, pred_labels):
        for k, v in ann.items():
            pred_entities.append(k)
    pred_counter = collections.Counter(pred_entities)
    for entity, freq in counter.most_common(1000):
        print('-------------------\n{}\n true : {}\n pred : {}'.format(entity, freq, pred_counter.get(entity, 0)))
    print(classification_report(labels, pred_labels))
    print(confusion_matrix(labels, pred_labels))
    return 'eval'


if __name__ == "__main__":
    app.run(host='localhost')
