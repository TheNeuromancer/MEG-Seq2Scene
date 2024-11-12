for sub in $(python configs/config.py)
do
	for cond in two_objects # localizer # one_object one_object  #
	do
		for th in 0 0.5 .55 # .58 # .59
		do
			# echo "python 08_dimensionality_analysis.py --localizer --auc_th $th -s $sub -c v33_config --train-cond $cond -w"
# --reconstruct_queries \"Complexity==0\" --reconstruct_queries \"Complexity==1\" --reconstruct_queries \"Complexity==2\" "

			echo "python 08_dimensionality_analysis.py --localizer --auc_th $th -s $sub -c v33_config --train-cond $cond --train-query \"Complexity==0\" -w"
			echo "python 08_dimensionality_analysis.py --localizer --auc_th $th -s $sub -c v33_config --train-cond $cond --train-query \"Complexity==1\" -w"
			echo "python 08_dimensionality_analysis.py --localizer --auc_th $th -s $sub -c v33_config --train-cond $cond --train-query \"Complexity==2\" -w"
		done
	done
done
