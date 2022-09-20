for sub in $(python configs/config.py)
do
	echo "python 06_same_different_cca.py -s $sub --equalize_events_same -c v23_config -w"
	echo "python 06_same_different_cca.py -s $sub --equalize_events_same -c v23_config -w --mirror_img"

	echo "python 06_same_different_cca.py -s $sub --equalize_events_same -c v23_config -w --windows '3,5'"
	echo "python 06_same_different_cca.py -s $sub --equalize_events_same -c v23_config -w --windows '3,5' --mirror_img"
done


