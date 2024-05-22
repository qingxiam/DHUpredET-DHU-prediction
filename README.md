# <span style="font-size:16px;">**DHUpredET: A Comparative Computational Approach for Identification of Dihydrouridine Modification Sites in RNA Sequence**</span>

## <span style="font-size:16px;">**Overview:**</span>


<span style="font-size:14px;">DHUpredET is a computational tool designed for the identification of dihydrouridine (DHU) modification sites in RNA sequences. This repository contains the implementation of DHUpredET as well as datasets collected from Stack-DHUpred.</span>

## <span style="font-size:16px;">**Features:**</span>

<span style="font-size:14px;">- Identification of DHU modification sites in RNA sequences
- Comparative computational approach for accurate predictions
- Utilizes machine learning techniques for classification
- Used numerous ML models on 9 feature extractions</span>



## <span style="font-size:16px;">**File Formats:**</span>

<span style="font-size:14px;">
All datasets are provided in FASTA format, which can have either a .txt or .fasta extension. 
Please go to the (https://en.wikipedia.org/wiki/FASTA_format) for more information.

## Workflow Diagram

![Workflow Diagram](https://drive.google.com/uc?export=view&id=1KvvGdqf4weDc5GH-IwFkCeuvXfLhw6nr)
**For more information please see the notebook**

## Proposed Architecture of the study
![DHUpredWT Model](https://drive.google.com/uc?export=view&id=1EyAQO9s1ektVqNoEnGH6k0-GNb6c4qgZ)

## Our DHUpredET model achieved significant improvements over previous approaches....

| Model                | Acc   | MCC    | Sen    | Spe    | AUC   | PR    |
|----------------------|-------|--------|--------|--------|-------|-------|
| Dpred [9]            | 0.68  | 0.384  | 0.508  | 0.853  | 0.703 | 0.706 |
| Stack-DHUpred [19]   | 0.778 | 0.567  | 0.725  | 0.830  | 0.871 | 0.893 |
| iDHU-Ensem [18]      | 0.574 | 0.152  | 0.689  | 0.459  | 0.574 | 0.541 |
| iRNAD [14]           | 0.721 | 0.489  | 0.508  | 0.934  | 0.721 | 0.696 |
| **DHUpredET**        | **0.8525** | **0.7053** | **0.8689** | **0.8361** | **0.9105** | **0.9147** |



## <span style="font-size:16px;">**Requirements:**</span>

<span style="font-size:14px;">
To run DHUpredET, ensure the following packages are installed:
- matplotlib==3.6.2
- numpy==1.24.4
- scikit-learn==1.1.3

You can install these packages using pip:

```bash

pip install matplotlib==3.6.2

pip install numpy==1.24.4

pip install scikit-learn==1.1.3





from google.colab import drive

drive.mount('/content/drive')





Collected dataset from Stack-DHUpred { published paper}

