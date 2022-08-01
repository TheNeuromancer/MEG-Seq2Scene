for sub in "all" #"v1" "v2" "goods" # $(python configs/config.py)
do
	echo "python 05_plot_decoding.py -w -s $sub --slices"
	echo "python 05_plot_decoding.py -w -s $sub --slices --ovr" 
	echo "python 05_plot_decoding.py -w -s $sub --slices --regression"
	# echo "python 05_plot_decoding.py -w -s $sub --slices --correlation"
done