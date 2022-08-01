for sub in $(python configs/config.py)
do
	echo "python 06_same_different_cca.py -s $sub --equalize_events_same -w"
	echo "python 06_same_different_cca.py -s $sub --equalize_events_same -w --mirror_img"
done


