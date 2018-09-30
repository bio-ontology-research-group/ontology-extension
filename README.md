# Combining lexical and context features for automatic ontology extension
Programming languages:
  * Python 3.6
  * Groovy
* We generate the word vectors for our large corpus (~ 58G) from [Europe PMC](http://europepmc.org/ftp/archive/v.2017.06/) using IBEX Cluster.  
* Using **generateMAP.groovy** to identify Infectious and Anatomical disease classes and their corresponding subclasses.  
* With **Diseases.py** we use the whole generated vectors to train ANN with different hidden layer sizes to classifiy wheather a term refer to a disease class or non-disease term.  
* With **inecAnato.py** we use ```2resINF.txt``` to train ANN to classifiy wheather a term refer to one of the Infectious disease sub-classes (bactiral, fungal, parastic or viral), ```2resANA.txt``` Anatomical disease sub-classes (12 different sub-classes) or ```2resboth.txt``` combining both classes.  
* The evaluation for each cases was done using F-score and AUC functions within **Diseases.py** and **inecAnato.py**
