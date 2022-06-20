# MEEG-classification 

Repository for epileptic spike classification based on [Transformer-based Spatial-Temporal Feature
Learning for EEG Decoding](https://arxiv.org/pdf/2106.11170.pdf).

## Architecture
![Screenshot 2022-06-20 at 3 57 54 PM](https://user-images.githubusercontent.com/64415312/174670350-f829cd5e-5281-4e06-8a3a-9157072800b0.png)

## Recover code
Clone this github repository by running in terminal `git clone -b ao/seizure_classification https://github.com/AmbroiseOdonnat/MEEG-Brainstorm.git`.  

## Virtual environment setup
First, make sure that a compatible version of Python 3 is installed on your system by running in terminal: `python3 --version`.  
It requires Python >= 3.6 and <3.10.  

With X being your python3 version, create virtual environment by running in terminal: `python3.X -m venv transformer_env`.   

Activate transformer_env by running in terminal: `source transformer_env/bin/activate`.  

Go to repository location. Install the requirements by running in terminal: `pip install -r requirements.txt`
