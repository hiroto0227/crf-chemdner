import pycrfsuite


class CRFTrainer:
    def __init__(self):
        self.trainer = pycrfsuite.Trainer()

    def train(self, filepath, features, labels):
        self.trainer.append(features, labels)
        self.trainer.set_params({
            'c1': 1.0,   # coefficient for L1 penalty
            'c2': 1e-3,  # coefficient for L2 penalty
            'max_iterations': 200,  # stop earlier
            # include transitions that are possible, but not observed
            'feature.possible_transitions': True
        })
        print('---------- Train Start -------------')
        self.trainer.train(filepath)
        print('---------- Train End -------------')
