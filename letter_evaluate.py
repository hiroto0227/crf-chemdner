from scripts import datasets
from scripts import transformer
from scripts import models
import config
from tqdm import tqdm
from termcolor import cprint


def ann_to_annfile(anns):
    """.ann形式に変換する。"""
    annfile_text = ''
    for ann in anns:
        entity = ''.join([k for k in ann.keys()])
        start = ann[entity][0]
        end = ann[entity][1]
        annfile_text += 'T1\tOutput {} {}\t{}\n'.format(start, end, entity)
    return annfile_text


if __name__ == '__main__':
    test_texts, test_anns, file_ids = datasets.test()
    tf = transformer.CharacterInvertTransformer()
    features = []
    true_labels = []
    for text, ann in tqdm(zip(test_texts, test_anns)):
        features.append(tf.text2features(text))
        true_labels.append(tf.text_anns2labels(text, ann))
    # load model
    model = models.CRFModel()
    model.load(config.model_root + 'character_invert')
    # predict label
    print('------ start predicting -------')
    pred_labels = [model.predict(feature) for feature in features]
    # convert Anns data format
    print('------ start evaluate --------')
    correct_count = 0
    all_true_count = 0
    all_pred_count = 0
    for file_id, test_text, pred_label, true_label in tqdm(zip(file_ids, test_texts, pred_labels, true_labels)):
        pred_anns = tf.text_labels2anns(test_text, pred_label)
        true_anns = tf.text_labels2anns(test_text, true_label)
        if pred_anns != true_anns:
            cprint('-' * 30, 'red')
            #print(true_anns)
            #print('-' * 10)
            #print(pred_anns)
        all_true_count += len(true_anns)
        all_pred_count += len(pred_anns)
        for p in pred_anns:
            for t in true_anns:
                if p.values() == t.values():
                    correct_count += 1
        #with open('./test_ann_files/{}.ann'.format(file_id), 'wt') as f:
        #    f.write(ann_to_annfile(pred_anns))
    print('全entity数 : {}'.format(all_true_count))
    print('予測entity数 : {}'.format(all_pred_count))
    print('正解entity数 : {}'.format(correct_count))
    precision = correct_count / all_pred_count
    recall = correct_count / all_true_count
    print('精度 : {}'.format(correct_count / all_pred_count))
    print('再現率 : {}'.format(correct_count / all_true_count))
    print('F値 : {}'.format(2 * precision * recall / (precision + recall)))
