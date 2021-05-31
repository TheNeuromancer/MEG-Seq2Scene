for sub in $(python configs/config.py)
do
	for cond in one_object two_objects # localizer
	do
		for dist_metric in correlation confusion
		do
			for rsa_metric in pearson regression # spearman 
			do
				echo "python 06_rsa.py -s $sub --train-cond $cond --rsa_metric $rsa_metric --distance_metric $dist_metric -w --xdawn "
				# -c v8config --filter \"Matching=='match'\" "
				# 
			done
		done
	done
done
