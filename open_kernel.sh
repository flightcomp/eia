#! /bin/bash
source ~/.virtualenvs/transformers/bin/activate
python -m ipykernel install --user --name=transformers --display-name "Python (transformers)"
deactivate
jupyter kernelspec list
