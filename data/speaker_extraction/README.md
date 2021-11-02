## Usage

Run: bash preprocess.sh


## Generated dataset file structure



	voxceleb2
	  └── orig
	    |── train     	# The original train set contains .mp4 video
	    └── test		# The original test set contains .mp4 video	
	  └── audio_clean	
	    |── train     	# The extrated train set contains .wav audio
	    └── test		# The extrated test set contains .wav audio	
	  └── audio_mixture
	    └──2_mix_min_800 # The simulated 2 speaker mixture contatins .wav audio	
	      |── train
	      |── val
	      └── test

