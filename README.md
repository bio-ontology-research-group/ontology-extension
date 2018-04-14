# ConceptRecognition_word2vec 
* We generate the word vectors for our large corpus (~ 58G) using IBEX Cluster.  
* Using **generateMAP.groovy** to identify Infectious and Anatomical disease classes and their corresponding subclasses.  
* With **Disease.py** we use the whole generated vectors to train SVM and ANN to classifiy wheather a term refer to a disease class or non-disease term.  
* With **Infectious-Anatomical.py** we use DiseaseVectors.txt to train SVM and ANN to classifiy wheather a term refer to one of the Infectious disease sub-classes (bactiral, fungal, parastic or viral) or Anatomical disease sub-classes (12 different sub-classes) or other disease class.  
* The evaluation for each cases was done using F-score and AUC functions within **Disease.py** and **Infectious-Anatomical.py**
