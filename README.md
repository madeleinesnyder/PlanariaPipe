# PlanariaPipe
Pipeline for processing raw .bin vidoes
If you want to regenerate the masks, you will need to create a conda envrionment to run SAM
1. Only if you want to regenerate the masks or anonymized videos: Run MN_1_DrawBoxes_MCellS.py. It will prompt you, enter the name of the .bin file you want to segment into worms (there are 6 worms in each video, so you draw 6 boxes one for each lane with a single worm). Something like: ./data/Raw_data/2025_10_15_10_20_58_trial_1_TC

2. Only if you want to regenerate the masks (or anonymized videos): Run MN_2_Make_JPGs.py. There are two ways to run this. The first is to set the following parameters at the beginning of the script: 

To get cropped and anonymized videos for hand-scoring:

LONG_OR_SHORT = 'short'
ANONYMIZATION = True
SEPARATE_CS_ON_OFF_BACKGROUNDS = True
CROPPED_VIDEOS = True

To get longform videos for further processing: 

LONG_OR_SHORT = 'long'
ANONYMIZATION = False
SEPARATE_CS_ON_OFF_BACKGROUNDS = True
CROPPED_VIDEOS = False

3. Only if you want to regenerate the masks: Run MN_3_SAM2_video_predictor_iterative.py

4. Run MN_4_Extract_Feature_AddingWheels.py to extract features from the FINAL masks. Set ASSUME_FINAL = True if you already have the masks. 

5. Run MN_5_Calculate_KL.py to calculate the KL divergence scores for the features.

6. Run MN_6_Plots_and_Stats_adding_pseudoconditioning.py to plot everything.
