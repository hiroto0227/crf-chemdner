import pycrfsuite
import datetime


class CRFModel:
    crf = None
    __crf_path__ = '../models/conll2002-esp.crfsuite'
    # __crf_path__ = '../models/crf{0:%Y-%m-%d_%H:%M:%S}.crfsuite'.format(datetime.datetime.now())

    def __init__(self):
        pass

    def fit(self, features, labels):
        """X_train, Y_trainはそれぞれの素性である。
        features = [{'token':token, '2-gram':2-gram-token}, ...]
        labels = ['B', 'M', 'E', 'O', 'O', ... 'O']

        feature : {'token':token, '2-gram':2-gram-token ...}
        label : 'B' or 'M' or 'E' or 'S' or 'O'
        """
        assert len(features) == len(labels), 'You must the same length betweens features and labels'
        self.trainer = pycrfsuite.Trainer(verbose=False)
        self.trainer.append(features, labels)
        self.trainer.set_params({
            'c1': 1.0,   # coefficient for L1 penalty
            'c2': 1e-3,  # coefficient for L2 penalty
            'max_iterations': 50,  # stop earlier
            # include transitions that are possible, but not observed
            'feature.possible_transitions': True
        })
        print('---------- Train Start -------------')
        self.trainer.train(self.__crf_path__)
        print('---------- Train End -------------')

    def predict(self, features):
        print('---------- Predict -------------')
        self.tagger = pycrfsuite.Tagger()
        self.tagger.open(self.__crf_path__)
        return self.tagger.tag(features)
