for sub in $(python configs/config.py)
do
	echo "python 00_detect_bad_channels.py -w -s $sub"
done