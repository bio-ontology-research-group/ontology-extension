# Combining lexical and context features for automatic ontology extension

This repository contains script which were used to build and train the prediction models together with the scripts for evaluating their performance.

# Data
Full-text PMC articles from [Europe PMC](http://europepmc.org/ftp/archive/v.2017.06/).

# Dependencies

To install python dependencies run: `pip install -r requirements.txt`

# Scripts

* Using `Reasoner.groovy` to identify Infectious and Anatomical disease classes and their corresponding subclasses. 
* Using `labelSynoExtraction.groovy` to extract ontology synonyms/labels.
* With `Diseases.py` we use the whole generated vectors to train ANN with different hidden layer sizes to classifiy wheather a term refer to a disease class or non-disease term.  
* With `inecAnato.py` we use ***2resINF.txt*** to train ANN to classifiy wheather a term refer to one of the Infectious disease sub-classes (bactiral, fungal, parastic or viral), ***2resANA.txt*** Anatomical disease sub-classes (12 different sub-classes) or ***2resboth.txt*** combining both classes.  
* The evaluation for each cases was done using F-score and AUC functions within `Diseases.py` and `inecAnato.py`

# Workflow

**STEP1.** Annotate Full-text PMC articles by employing [Whatizit](https://github.com/bio-ontology-research-group/whatizit) with disease names.

**STEP2.** Generate the word embeddings for the annotated text using word2vec.

**STEP3.** Run `Diseases.py` script with specifiying the file name containing the resulted embeddings from STEP2 (***CleanAllVectors.txt*** in our case) + ***DiseasVectors.txt*** in order to predict wheather a word is a disease of other.

**STEP4.** Run `inecAnato.py` script with specifiying the file name containing the embeddings ***2resINF.txt*** or ***2resANA.txt*** or ***2resboth.txt*** in order to predict a new sub-classes within infectious disease, anatomical disease or both of them.

# Final notes
For any comments or help needed with how to run the scripts, please send an email to: sara.althubaiti@kaust.edu.sa
