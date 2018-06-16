from scripts import datasets
from scripts import transformer
from scripts import models
import config
from tqdm import tqdm


if __name__ == '__main__':
    test_texts, test_anns = datasets.test()
    tf = transformer.LetterLevelTransformer()
    features = []
    true_labels = []
    for text, ann in tqdm(zip(test_texts, test_anns)):
        features.append(tf.convertTextToFeatures(text))
        true_labels.append(tf.convertAnnsToLabels(ann, text))
    # load model
    model = models.CRFModel()
    model.load(config.model_root + 'crf_suite_letter')
    # predict label
    print('------ start predicting -------')
    pred_labels = [model.predict(feature) for feature in features]
    # convert Anns data format
    print('------ start evaluate --------')
    correct_count = 0
    all_true_count = 0
    all_pred_count = 0
    for test_text, pred_label, true_label in tqdm(zip(test_texts, pred_labels, true_labels)):
        pred_anns = tf.convertLabelsToAnn(test_text, pred_label)
        true_anns = tf.convertLabelsToAnn(test_text, true_label)
        all_true_count += len(true_anns)
        all_pred_count += len(pred_anns)
        for p in pred_anns:
            for t in true_anns:
                if p == t:
                    correct_count += 1
    print('正解データ数 : {}'.format(all_true_count))
    print('予測データ数 : {}'.format(all_pred_count))
    print('正解数 : {}'.format(correct_count))
    print('精度 : {}'.format(correct_count / all_pred_count))
    print('再現率 : {}'.format(correct_count / all_true_count))
