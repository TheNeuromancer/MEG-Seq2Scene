for sub in $(python configs/config.py) "all" "v1" "v2" "goods"
do
	echo "python 05_plot_decoding.py -w -s $sub"
	echo "python 05_plot_decoding.py -w -s $sub --ovr"
done