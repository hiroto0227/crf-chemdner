from scripts import datasets
from scripts.transformer import Transformer
from scripts import trainer
import config
from tqdm import tqdm

if __name__ == '__main__':
    train_texts, train_anns = datasets.train()
    tf = Transformer()
    features = []
    labels = []
    for text, ann in tqdm(zip(train_texts, train_anns)):
        features.extend(tf.convertTextToNgram(text))
        labels.extend(tf.convertAnnsToLabels(ann, len(text)))
    assert len(features) == len(labels), 'Not the same length, features and labels'
    trainer = trainer.CRFTrainer()
    trainer.train(config.model_root + 'crf_suite_v1', features, labels)
