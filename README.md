# crf-chemdner

## Abstract
CRF model for chemdner

## Directory Structure
```
├── Procfile # for flask deploy
├── README.md
├── app.py # model api
├── config.py # for dependency 
├── evaluate.py
├── experiment.py
├── requirements.txt
├── runtime.txt # for flask deploy
├── scripts
│   ├── __pycache__
│   │   ├── convertCorpus2Features.cpython-36.pyc
│   │   ├── datasets.cpython-36.pyc
│   │   ├── featurize.cpython-36.pyc
│   │   ├── models.cpython-36.pyc
│   │   ├── trainer.cpython-36.pyc
│   │   └── transformer.cpython-36.pyc
│   ├── datasets.py
│   ├── models.py
│   ├── trainer.py
│   └── transformer.py
├── storage # git ignore
    ├── datas
    │   ├── train
    │   ├── valid
    │   └── test
    ├── models
    │   └── crf_suite_v1
    ├── rawdatas
    └── transformers
```
