for sub in $(python configs/config.py)
do
	for cond in localizer one_object two_objects
	do
		for dist_metric in correlation confusion
		do
			for rsa_metric in spearman pearson regression
			do
				echo "python 06_rsa.py -w -s $sub --train-cond $cond --rsa_metric $rsa_metric --distance_metric $dist_metric "
			done
		done
	done
done
