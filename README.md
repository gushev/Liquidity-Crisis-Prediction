# Liquidity Crisis Prediction

A simple predictor on liquidity crisis, that works with a given dataset (data/data_company.trn).

The projects is done as part of a Data Science class, that took place in Plovdiv University "Paisii Hilendarski".

## Getting Started

The project uses has a virtual environment set up in place (venv), but it can also be installed using the requirements.txt file.

### Prerequisites

I've developed it on Python 3.6.7 and that's the python version that the virtual env is using, but it should work on any Python 3 version. 
Use the requirements.txt for any other dependencies.

### Installing

Running it with the virtual environment, that is in the repository:

```
$ source venv/bin/activate
(venv) $ python main.py
```

Or creating your own virtual environment with the python version that you have.

```
$ virtualenv -p python3 venv
$ source venv/bin/activate 
(venv) $ pip install -r requirements.txt --no-index
(venv) $ python main.py
```