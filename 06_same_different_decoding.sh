for sub in $(python configs/config.py)
do
	echo "python 06_same_different_decoding.py -s $sub --equalize_events_same -w --windows '3,5' -c v23_config "
	echo "python 06_same_different_decoding.py -s $sub --equalize_events_same -w --mirror_img  --windows '3,5' -c v23_config "

	echo "python 06_same_different_decoding.py -s $sub --equalize_events_same -w -c v23_config "
	echo "python 06_same_different_decoding.py -s $sub --equalize_events_same -w --mirror_img -c v23_config "

	# reduc dim
	echo "python 06_same_different_decoding.py -s $sub --equalize_events_same -w --windows '3,5' --reduc_dim_same 0.99 -c v23_config "
	echo "python 06_same_different_decoding.py -s $sub --equalize_events_same -w --mirror_img  --windows '3,5' --reduc_dim_same 0.99 -c v23_config "

	echo "python 06_same_different_decoding.py -s $sub --equalize_events_same -w --reduc_dim_same 0.99 -c v23_config "
	echo "python 06_same_different_decoding.py -s $sub --equalize_events_same -w --mirror_img --reduc_dim_same 0.99 -c v23_config "

done


