for sub in $(python configs/config.py)
do
	echo "python 01_raw2epo.py -w -s $sub --plot -c v22_config"
done