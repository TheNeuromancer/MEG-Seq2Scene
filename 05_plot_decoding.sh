for sub in $(python configs/config.py) "all"
do
	echo "python 05_plot_decoding.py -w -s $sub"
done