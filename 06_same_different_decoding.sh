for sub in $(python configs/config.py)
do
	echo "python 06_same_different_decoding.py -s $sub --equalize_events_same -w --windows '3,5'"
	echo "python 06_same_different_decoding.py -s $sub --equalize_events_same -w --mirror_img  --windows '3,5'"

	echo "python 06_same_different_decoding.py -s $sub --equalize_events_same -w"
	echo "python 06_same_different_decoding.py -s $sub --equalize_events_same -w --mirror_img"

	# reduc dim
	echo "python 06_same_different_decoding.py -s $sub --equalize_events_same -w --windows '3,5' --reduc_dim_same 0.99"
	echo "python 06_same_different_decoding.py -s $sub --equalize_events_same -w --mirror_img  --windows '3,5' --reduc_dim_same 0.99"

	echo "python 06_same_different_decoding.py -s $sub --equalize_events_same -w --reduc_dim_same 0.99"
	echo "python 06_same_different_decoding.py -s $sub --equalize_events_same -w --mirror_img --reduc_dim_same 0.99"

done


