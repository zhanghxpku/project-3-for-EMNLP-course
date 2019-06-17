# project-3-for-EMNLP-course

## Requirements
- Python 2.7
- Tensorflow 1.12.0
- NumPy 1.16.3
- *Python 3.6 (to preprocesse the data or transform inferring output to required format)*

## Implementation Instructions
0. Data preprocessing
- Datasets should be put in **data** folder
- GloVE embedding should be put in **data/embedding** folder
- python3 new_processing.py

1. To train the model on training set
- python2 run.py semantic_parsing_model.yaml -action train

2. To evaluate the model on development set
- python2 run.py semantic_parsing_model.yaml -action eval

3. To produce the prediction on test set
- python2 run.py semantic_parsing_model.yaml -action infer

4. Transform inferring results to required format
- results in **results/semantic_parsing** folder
- python3 results/trans_format.py