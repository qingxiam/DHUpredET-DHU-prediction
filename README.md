DHUpredET: A Comparative Computational Approach for Identification of Dihydrouridine Modification Sites in RNA Sequence
[1]. Read File:
All the datasets file are in FASTA format which can be with .txt or .fasta extension. E.g. anyName.txt or anyName.fasta. Please know more about the FASTA file format by clicking here!.

How to run packages: 
Requirments: 
!pip install matplotlib==3.6.2
!pip install numpy==1.24.4
!pip install scikit-learn==1.1.3
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score, recall_score, f1_score
from google.colab import drive
drive.mount('/content/drive')

Collected dataset from Stack-DHUpred { published paper}
