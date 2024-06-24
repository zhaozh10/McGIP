## Repository Structure

This Folder contains following contents:

1. **BYOL-Gaze.py**: we conduct experiments under mmselfsup 0.x environments, this is the modified gaze-aided BYOL. The main difference is shown in self.\_create_buffer(N, idx_list)

2. **STGM_Preprocess.py**: The function gaze_cluster illustrates the implementation of (1) temporal embedding and (2) clustering shown in Fig. 2.

3. **STGM_Similarity.py**: This program contains the implementation of Hu-Moments and the process of computing STGM similarity. Specifically, extract_cluster_mu_Hu(idx,namelist) is used to extract Hu-Moments vectors from each sample's gaze heatmaps (recall that different gaze sequences can be clustered into different number of clusters). The function accelerated_dtw is adopted in line 102 to compute **difference** between two STGMs.
