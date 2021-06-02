for sub in $(python configs/config.py)
do
	for cond in one_object two_objects # localizer
	do
		for dist_metric in confusion correlation
		do
			for rsa_metric in regression RF_regression # spearman pearson
			do
				for xdawn_str in '' # --xdawn
				do
					echo "python 06_rsa.py -s $sub --train-cond $cond --distance_metric $dist_metric -w $xdawn_str --rsa_metric pearson --rsa_metric regression "
					# -c v8config --filter \"Matching=='match'\" "
					# 
				done
			done
		done
	done
done
