for sub in $(python configs/500hz_config.py)
do
	echo "python 12_pac.py -s $sub -w -c 500hz_config --train-query Complexity --windows '4,5' "
done

