from scripts import datasets
from scripts import transformer
from scripts import models
import config
from tqdm import tqdm
from termcolor import cprint

if __name__ == '__main__':
    test_texts, test_anns, _ = datasets.test()
    tf = transformer.CharacterInvertTransformer()
    features = []
    true_labels = []
    converted_anns = []
    for text, ann in zip(test_texts, test_anns):
        label = tf.text_anns2labels(text, ann)
        converted_ann = tf.text_labels2anns(text, label)
        converted_anns.append(converted_ann)
        if ann and converted_ann and ann != converted_ann:
            cprint('=' * 70, 'red')
            print(text)
            print('========')
            print(''.join([feature['a_5'] for feature in tf.text2features(text)]))
            print('========')
            print(label)
            print('=========')
            print(ann)
            print('=========')
            print(converted_ann)
    print(sum([len(a) for a in test_anns]))
    print(sum([len(a) for a in converted_anns]))
