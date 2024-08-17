# Mushroom Classifier - Poisonous or Edible 

## Overview
Basic model using XgBoost to predict whether a mushroom is safe to eat.

## Installation
Step-by-step instructions on how to install the necessary dependencies on mac.

0. **Create and activate a virtual environment**
   Create using:
   ```bash
   python3 -m venv name_of_virtual_env
   ```
   Activate using:
   ```bash
   source name_of_virtual_env/bin/activate
   ```
   
1. **Clone the repository**  
   Run the following command to clone the repository to your local machine:
   ```bash
   git clone https://github.com/akashsp7/Mushroom-classification.git
   ```
2. **Install pred_model package in your virtual environment**
   Make sure to be inside the venv before this step to not have this package in your global environment.
   ```bash
   pip install git+https://github.com/akashsp7/Mushroom-classification.git
   ```
3. **Training**
   Open terminal inside pred_model folder and run training_pipeline.py.
   ```bash
   python3 training_pipeline.py
   ```
4. **Prediction**
   Open terminal inside pred_model folder and run predict.py.
   ```bash
   python3 predict.py
   ```
   
## Config
Since config is inside the package itself, changing it requires building the package again. To create different configuration, skip step 2 and add the 
pred_model path using sys.
```python
import sys

ROOT = '<Path to pred_model>'
sys.append(ROOT)
   
