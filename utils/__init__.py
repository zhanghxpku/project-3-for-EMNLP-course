#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os
import logging
import logging.config


initialized = False
def initialize_logging(out, level='DEBUG'):
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'default': {
                'level':level,
                'class':'logging.StreamHandler',
                'stream': out
            },
        },
        'loggers': {
            '': {
                'handlers': ['default'],
                'level': level,
                'propagate': True
            }
        }
    })
    initialized = True

from python_dataset import PythonDataset
def get_dataset(config):
    return PythonDataset(config)

from python_estimator import PythonEstimator
def get_estimator(config, model):
    logging.debug('estimator type: %s', config.type) 
    return PythonEstimator(config, model)

def main():
    pass

if __name__ == '__main__':
    main()

