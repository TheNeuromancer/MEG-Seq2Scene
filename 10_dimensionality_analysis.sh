for sub in $(python configs/config.py)
do
	for cond in one_object two_objects # localizer
	do
		for xdawn_str in '' # --xdawn
		do
			echo "python 10_dimensionality_analysis.py -s $sub --train-cond $cond -w $xdawn_str "
		done
	done
done
