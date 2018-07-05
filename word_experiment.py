from scripts import datasets
from scripts import transformer
from scripts import trainer
import config
from tqdm import tqdm

if __name__ == '__main__':
    train_texts, train_anns = datasets.train()
    tf = transformer.WordLevelTransformer()
    # word level
    features = []
    labels = []
    for text, ann in tqdm(zip(train_texts, train_anns)):
        features.extend(tf.text2features(text))
        labels.extend(tf.text_ann2labels(text, ann))
    print('training start !!!')
    word_trainer = trainer.CRFTrainer()
    word_trainer.train(config.model_root + 'crf_suite_word', features, labels)
    print('training end !!!')
