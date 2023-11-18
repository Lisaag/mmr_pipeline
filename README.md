# Multimedia Retrieval Pipeline

## Controls

The MMR pipeline mesh viewer controls are listed below:

- (q) Quit the mesh viewer
- (esc) Show next mesh
- (p/P) Show points and increase / decrease point size
- (w) Show wireframe mesh
- (s) Show solid mesh


1. To install all neccessary packages, run the following command:
pip install -r requirements.txt

2. Extract and add analyze-results to the root folder python-proj




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