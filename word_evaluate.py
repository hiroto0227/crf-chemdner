from scripts import datasets
from scripts.transformer import Transformer
from scripts import models
import config
from tqdm import tqdm

if __name__ == '__main__':
    test_texts, test_anns = datasets.test()
    tf = Transformer()
    features = []
    true_labels = []
    for text, ann in tqdm(zip(test_texts, test_anns)):
        features.extend(tf.convertTextToWordNgram(text))
        true_labels.extend(tf.convertAnnsToWordLabels(ann, text))
    # load model
    model = models.CRFModel()
    model.load(config.model_root + 'crf_suite_word')
    # predict label
    print('------ start predicting -------')
    pred_labels = model.predict(features)
    # convert Anns data format
    texts = ' '.join([d['1g'] for d in features])
    print([d['1g'] for d in features])
    print(texts)
    print(len(texts.split(' ')))
    print('---------')
    print(len(pred_labels))
    pred_anns = tf.convertWordLabelsToAnn(texts, pred_labels)
    true_anns = tf.convertWordLabelsToAnn(texts, true_labels)
    # print results
    print('------- start evaluating ---------')
    correct_count = 0
    for p in tqdm(pred_anns):
        for t in true_anns:
            if p == t:
                correct_count += 1
    print('全データ数 : {}'.format(len(true_anns)))
    print('予測データ数 : {}'.format(len(pred_anns)))
    print('正解数 : {}'.format(correct_count))
    print('精度 : {}'.format(correct_count / len(pred_anns)))
    print('再現率 : {}'.format(correct_count / len(true_anns)))
