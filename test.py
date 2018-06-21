from scripts import datasets
from scripts import transformer
from scripts import models
import config
from tqdm import tqdm
from termcolor import cprint

if __name__ == '__main__':
    test_texts, test_anns = datasets.test()
    tf = transformer.LetterLevelTransformer()
    features = []
    true_labels = []
    converted_anns = []
    for text, ann in zip(test_texts, test_anns):
        label = tf.convertAnnsToLabels(ann, text)
        converted_ann = tf.convertLabelsToAnn(text, label)
        converted_anns.append(converted_ann)
        if ann and converted_ann and ann != converted_ann:
            cprint('=' * 70, 'red')
            print(text)
            print('========')
            print(label)
            print('=========')
            print(ann)
            print('=========')
            print(converted_ann)
    print(sum([len(a) for a in test_anns]))
    print(sum([len(a) for a in converted_anns]))
