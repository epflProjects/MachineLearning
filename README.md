# Machine Learning - Finding the Higgs boson

This software allows to classify data detected in a particles collider into 2 groups depending on the detection or not of the Higgs boson.

## Input Data Form

The input data has to be a CSV file with the following attributes : `Id`, `Predict`, `ion`, `DER_mass_MMC`, `DER_mass_transverse_met_lep`, `DER_mass_vis`, `DER_pt_h`, `DER_deltaeta_jet_jet`, `DER_mass_jet_jet`, `DER_prodeta_jet_jet`, `DER_deltar_tau_lep`, `DER_pt_tot`, `DER_sum_pt`, `DER_pt_ratio_lep_tau`, `DER_met_phi_centrality`, `DER_lep_eta_centrality`, `PRI_tau_pt`, `PRI_tau_eta`, `PRI_tau_phi`, `PRI_lep_pt`, `PRI_lep_eta`, `PRI_lep_phi`, `PRI_met`, `PRI_met_phi`, `PRI_met_sumet`, `PRI_jet_num`, `PRI_jet_leading_pt`, `PRI_jet_leading_eta`, `PRI_jet_leading_phi`, `PRI_jet_subleading_pt`, `PRI_jet_subleading_eta`, `PRI_jet_subleading_phi`, `PRI_jet_all_pt` 

## Output Data Form
The program outputs a CSV containing 2 columns. 
- `Id` : the id of the data
- `Prediction` : `1` if the detected data is a Higgs boson, `-1` if it isn't.

## How to run
Simply run `python3 run.py` once located in the folder containing `run.py`.

### Authors 
Arnaud Pannatier, Bastian Nanchen, Matteo Giorla

_EPFL Machine Learning CS-433 Course 2017_
