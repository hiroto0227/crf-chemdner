import pycrfsuite


class CRFModel:

    def __init__(self):
        self.crf_tagger = pycrfsuite.Tagger()

    def load(self, filepath):
        self.crf_tagger.open(filepath)

    def predict(self, features):
        return self.crf_tagger.tag(features)
