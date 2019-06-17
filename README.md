# project-3-for-EMNLP-course

## Requirements
- Python 2.7
- Tensorflow 1.12.0
- NumPy
- Python 3.6 (to preprocesse the data or transform inferring output to required format)

## Implementation Instructions
0. Data preprocessing
- python3 new_processing.py

1. To train the model on training set
- python2 run.py semantic_parsing_model.yaml -action train

2. To evaluate the model on development set
- python2 run.py semantic_parsing_model.yaml -action eval

3. To produce the prediction on test set
- python2 run.py semantic_parsing_model.yaml -action infer

4. Transform inferring results to required format
- python3 results/trans_format.py