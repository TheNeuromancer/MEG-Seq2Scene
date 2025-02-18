for sub in $(python configs/500hz_config.py)
do
	echo "python 01_raw2epo.py -w -s $sub --plot -c 500hz_config"
done
