for sub in $(python configs/config.py)
do
	for cond in two_objects # localizer # one_object 
	do
		for xdawn_str in '' # --xdawn
		do
			echo "python 08_dimensionality_analysis.py -s $sub --train-cond $cond -w $xdawn_str -c v22_config \
--reconstruct_queries \"Complexity==0\" --reconstruct_queries \"Complexity==1\" --reconstruct_queries \"Complexity==2\" "

			echo "python 08_dimensionality_analysis.py -s $sub --train-cond $cond --train-query \"Complexity==0\" -w $xdawn_str -c v22_config "
			echo "python 08_dimensionality_analysis.py -s $sub --train-cond $cond --train-query \"Complexity==1\" -w $xdawn_str -c v22_config "
			echo "python 08_dimensionality_analysis.py -s $sub --train-cond $cond --train-query \"Complexity==2\" -w $xdawn_str -c v22_config "
		done
	done
done
