for sub in $(python configs/config.py)
do
	for cond in two_objects # localizer # one_object 
	do
		for xdawn_str in '' # --xdawn
		do
			echo "python 08_dimensionality_analysis.py -s $sub --train-cond $cond -w $xdawn_str --micro_ave 2 --max_trials 300"

			echo "python 08_dimensionality_analysis.py -s $sub --train-cond $cond --train-query \"Complexity==0\" -w $xdawn_str --micro_ave 2 --max_trials 300"
			echo "python 08_dimensionality_analysis.py -s $sub --train-cond $cond --train-query \"Complexity==1\" -w $xdawn_str --micro_ave 2 --max_trials 300"
			echo "python 08_dimensionality_analysis.py -s $sub --train-cond $cond --train-query \"Complexity==2\" -w $xdawn_str --micro_ave 2 --max_trials 300"
		done
	done
done
