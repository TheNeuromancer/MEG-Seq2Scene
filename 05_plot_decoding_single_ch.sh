for sub in "all" "v1" "v2" "goods"
do
	echo "python 05_plot_decoding_single_ch.py -w -s $sub"
	echo "python 05_plot_decoding_single_ch.py -w -s $sub --regress"
done