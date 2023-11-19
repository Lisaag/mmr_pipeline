# Multimedia Retrieval Pipeline

## Controls
!!Make sure the project is opened with the root folder python-proj/!!

1. To install all neccessary packages, run the following commands:
pip install -r requirements.txt
pip install -U scikit-learn scipy
python -m pip install plotly

2. 
To run the querying app, run the following command (make sure k=6 in settings.py):
streamlit run mmr_pipeline/__query__.py

To run the app shown during my presentation (tsne plot, query shape by clicking a datapoint)
NOTE-> Might take a while (+-3 minutes) to generate tsne:
python mmr_pipeline/demo.py

To show tsne plot, run the following command;
NOTE-> Might take a while (+-3 minutes) to generate tsne
python mmr_pipeline/demo_tsne.py


All csv files and figures are in the folder analyze-results/
The original shape database can be found in data/original
The normalized shape database can be found in data/normalized