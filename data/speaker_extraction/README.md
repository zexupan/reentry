## Usage

Run: bash preprocess.sh


## Generated dataset file structure



	voxceleb2/
	  └── orig/
	    |── train/     	# The original train set contains .mp4 video
	    └── test/		# The original test set contains .mp4 video	
	  └── audio_clean/	
	    |── train/     	# (new) The extrated train set contains .wav audio
	    └── test/		# (new) The extrated test set contains .wav audio	
	  └── audio_mixture/
	    └──2_mix_min_800/ 	# (new) The simulated 2 speaker mixture contatins .wav audio	
	      |── train/
	      |── val/
	      |── test/
	      └──mixture_data_list_2mix.csv 	# The list of the simulated speech mixtures


