from scripts import datasets
from scripts.transformer import Transformer
from scripts import trainer
import config
from tqdm import tqdm

if __name__ == '__main__':
    train_texts, train_anns = datasets.train()
    tf = Transformer()
    # word level
    features = []
    labels = []
    for text, ann in tqdm(zip(train_texts, train_anns)):
        print(text)
        print('------------')
        print(ann)
        print('-----------')
        print(tf.convertTextToWordNgram(text))
        print('------------')
        features.extend(tf.convertTextToWordNgram(text))
        print(tf.convertAnnsToWordLabels(ann, text))
        print('------------')
        labels.extend(tf.convertAnnsToWordLabels(ann, text))
    print('training start !!!')
    word_trainer = trainer.CRFTrainer()
    word_trainer.train(config.model_root + 'crf_suite_word', features, labels)
    print('training end !!!')
