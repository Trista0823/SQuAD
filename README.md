# SQuAD

This project is for the final project for Stanford's CS224n Natural Language Processing with Deep Learning. In my project, I build question answering systems for the Standford Question Answering Dataset(SQuAD) 2.0. I explore two end-to-end models: the baseline BiDAF network and QANet, a non-recurrent model fully based on convolution and self-attention. 

## Main Accomplishment:
- Improve teh baseline BiDAF model by introducing character embeddings.
- Re-implement the QANet model from scratch.

## To run the code in this repo:
- setup the local environment using:
  - conda env create -f environment.yml
  - source activate squad
- run python setup.py to get teh data in the right folders
- run train.py --name
