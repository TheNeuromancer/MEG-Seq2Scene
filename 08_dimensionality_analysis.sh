for sub in $(python configs/config.py)
do
	for cond in one_object  #two_objects # localizer # one_object 
	do
		for th in .3 0.5 .52 .54 .55 .57 .59
		do
			echo "python 08_dimensionality_analysis.py --localizer --auc_th $th -s $sub --detrend -c v29_config --train-cond $cond -w"
# --reconstruct_queries \"Complexity==0\" --reconstruct_queries \"Complexity==1\" --reconstruct_queries \"Complexity==2\" "

			# echo "python 08_dimensionality_analysis.py --localizer --auc_th $th -s $sub --detrend -c v29_config --train-cond $cond --train-query \"Complexity==0\" -w"
			# echo "python 08_dimensionality_analysis.py --localizer --auc_th $th -s $sub --detrend -c v29_config --train-cond $cond --train-query \"Complexity==1\" -w"
			# echo "python 08_dimensionality_analysis.py --localizer --auc_th $th -s $sub --detrend -c v29_config --train-cond $cond --train-query \"Complexity==2\" -w"
		done
	done
done
