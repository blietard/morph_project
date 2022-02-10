## NLP: Sigmorphon - task 0
#### By Bastien Lietard, Tom Rivero & Marc-André Sergiel

### Installation

#### Required packages

- `numpy`
- `pandas`
- `pyenchant`
- `scipy`

#### Data

You can pull the `task0-data` repository wherever you want, you will just have to adjust the paths in the notebooks.  
We have placed it in the main folder of the repository.

### Structure of the repository

- `deprecated/`: contains the notebooks and Python files on which we have explored the data
- `modules/`: contains each sub-model of our non-neural model (Stemmer, NgramsExtractor, Assembler)
- `outputs/`: contains the files with the predictions of our non-neural model for the necessary languages
- `evaluation.ipynb`: the file that is used to produce a .out file to compute the accuracy and mean Levenshtein distance with the ground truth
- `grid-search.ipynb`: the file that is used to choose the best hyper-parameters by using the validation sets (.dev)
