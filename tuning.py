from scripts import datasets
from scripts import transformer
from scripts import trainer
from scripts import models
from tqdm import tqdm
import config


if __name__ == '__main__':
    tuned_parameters = [
        {'algolithm': 'lbfgs', 'c1': 1e-3, 'c2': 1e-3, 'feature.minfreq': 3, 'feature.possible_transitions': 1},
        {'algolithm': 'lbfgs', 'c1': 1e-3, 'c2': 1e-3, 'feature.minfreq': 5, 'feature.possible_transitions': 1},
        {'algolithm': 'lbfgs', 'c1': 1e-3, 'c2': 1e-3, 'feature.minfreq': 10, 'feature.possible_transitions': 1},
        {'algolithm': 'lbfgs', 'c1': 1, 'c2': 1e-3, 'feature.minfreq': 0, 'feature.possible_transitions': 1},
        {'algolithm': 'lbfgs', 'c1': 1e-2, 'c2': 1e-3, 'feature.minfreq': 0, 'feature.possible_transitions': 1},
        {'algolithm': 'lbfgs', 'c1': 1e-5, 'c2': 1e-3, 'feature.minfreq': 0, 'feature.possible_transitions': 1},
        {'algolithm': 'lbfgs', 'c1': 0, 'c2': 1, 'feature.minfreq': 0, 'feature.possible_transitions': 1},
        {'algolithm': 'lbfgs', 'c1': 0, 'c2': 1e-2, 'feature.minfreq': 0, 'feature.possible_transitions': 1},
        {'algolithm': 'lbfgs', 'c1': 0, 'c2': 1e-5, 'feature.minfreq': 0, 'feature.possible_transitions': 1},
    ]
    train_texts, train_anns = datasets.train()
    valid_texts, valid_anns = datasets.valid()
    tf = transformer.WordLevelTransformer()
    train_features = []
    train_labels = []
    for text, ann in tqdm(zip(train_texts, train_anns)):
        train_features.extend(tf.convertTextToFeatures(text))
        train_labels.extend(tf.convertAnnsToLabels(ann, text))
    valid_features = []
    valid_labels = []
    for text, ann in tqdm(zip(valid_texts, valid_anns)):
        valid_features.append(tf.convertTextToFeatures(text))
        valid_labels.append(tf.convertAnnsToLabels(ann, text))
    for i, parameter in enumerate(tuned_parameters):
        print('-----------trian_start! ---------------\n {}'.format(parameter))
        crf_trainer = trainer.CRFTrainer()
        crf_trainer.set_params(parameter)
        model_name = 'word_{}'.format('_'.join(map(str, parameter.values())))
        crf_trainer.train(config.model_root + model_name, train_features, train_labels)
        print('------------training end!!!-------------')
        model = models.CRFModel()
        model.load(config.model_root + model_name)
        print('-----------start predict ---------------')
        pred_labels = [model.predict(feature) for feature in valid_features]
        correct_count = 0
        all_true_count = 0
        all_pred_count = 0
        for valid_text, pred_label, valid_label in tqdm(zip(valid_texts, pred_labels, valid_labels)):
            pred_anns = tf.convertLabelsToAnn(valid_text, pred_label)
            true_anns = tf.convertLabelsToAnn(valid_text, valid_label)
            all_true_count += len(true_anns)
            all_pred_count += len(pred_anns)
            for p in pred_anns:
                for t in true_anns:
                    if p == t:
                        correct_count += 1
        precision = correct_count / all_pred_count
        recall = correct_count / all_true_count
        f_score = 2 * precision * recall / (precision + recall)
        print('精度:{}\n再現率:{}\nF値{}\n'.format(precision, recall, f_score))
