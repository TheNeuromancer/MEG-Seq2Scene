for sub in "all" "v1" "v2" "goods" # $(python configs/config.py)
do
	echo "python 05_plot_decoding.py -w -s $sub"
	echo "python 05_plot_decoding.py -w -s $sub --ovr"
done