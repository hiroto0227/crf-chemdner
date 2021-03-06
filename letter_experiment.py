from scripts import datasets
from scripts import transformer
from scripts import trainer
import config
from tqdm import tqdm

if __name__ == '__main__':
    train_texts, train_anns = datasets.train()
    tf = transformer.CharacterInvertTransformer()
    features = []
    labels = []
    for text, ann in tqdm(zip(train_texts, train_anns)):
        features.extend(tf.text2features(text))
        labels.extend(tf.text_anns2labels(text, ann))
    assert len(features) == len(labels), 'Not the same length, features and labels'
    print('training start!!!')
    letter_trainer = trainer.CRFTrainer()
    letter_trainer.train(config.model_root + 'character_invert', features, labels)
    print('training end !!!')
