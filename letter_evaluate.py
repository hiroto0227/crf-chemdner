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
    lens = 0
    for text, ann in tqdm(zip(test_texts, test_anns)):
        lens += len(ann)
        features.extend(tf.convertTextToLetterNgram(text))
        true_labels.extend(tf.convertAnnsToLetterLabels(ann, len(text)))
    # load model
    model = models.CRFModel()
    model.load(config.model_root + 'crf_suite_letter')
    # predict label
    print('------ start predicting -------')
    pred_labels = model.predict(features)
    # convert Anns data format
    texts = ''.join([d['a'] for d in features])
    pred_anns = tf.convertLetterLabelsToAnn(texts, pred_labels)
    true_anns = tf.convertLetterLabelsToAnn(texts, true_labels)
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
