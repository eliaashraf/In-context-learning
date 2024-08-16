# In-context-learning

This repository contains a Python-based solution for extracting architectural knowledge (AK) or keywords from emails. The process involves two main steps:

Fine-tuning GPT-2: A fine-tuning script is provided to train a GPT-2 model on a dataset of emails labeled with AKs.
AK Extraction: Once the model is fine-tuned, a Python script leverages few-shot learning to extract AKs from new emails.

Repository Structure:
- data: Contains the dataset used for fine-tuning the GPT-2 model. 
- fine_tuning: Contains the script for fine-tuning the GPT-2 model. 
- incontext_learning: Contains the script for extracting AKs from emails using the fine-tuned model.

## Installation

1. Clone the repository: ```git clone https://github.com/eliaashraf/In-context-learning.git```
2. Install required libraries: ```pip install -r requirements.txt```
3. Run time fine tuning script: `python fine_tuning.py`
4. Run the AK Extraction script: `python incontext_learning.py`
   
