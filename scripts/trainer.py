import pycrfsuite


class CRFTrainer:
    def __init__(self):
        self.trainer = pycrfsuite.Trainer(verbose=False)
        self.trainer.set_params({'feature.possible_transitions': 1})

    def train(self, filepath, features, labels):
        self.trainer.append(features, labels)
        self.trainer.set_params({
            'c1': 1.0,   # coefficient for L1 penalty
            'c2': 1e-3,  # coefficient for L2 penalty
            #'max_iterations': 200,  # stop earlier
            # include transitions that are possible, but not observed
            'feature.possible_transitions': True
        })
        print('---------- Train Start -------------')
        self.trainer.train(filepath)
        print('---------- Train End -------------')

    def set_params(self, params):
        """http://www.chokkan.org/software/crfsuite/manual.html#idp8849121424
        algolithm: lbfgs Gradient descent using the L-BFGS method
                   l2sgd Stochastic Gradient Descent with L2 regularization term
                   ap    Averaged Perceptron
                   pa    Passive Aggressive (PA)
                   arow  Adaptive Regularization Of Weight Vector (AROW)
        c1 : L1正規化項の重み(default=0)
        c2 : L2正規化項の重み(default=1)
        max_iterations : 何回学習するか(default=MAX)
        featrue.possible_transitions : training_dataにないものを1
        feature.minfreq : cutoffの頻度 (default=0)
        """
        if 'algolithm' in params.keys():
            self.trainer.select(params['algolithm'])
        self.params = {k: v for k, v in params.items() if k != 'algolithm'}
