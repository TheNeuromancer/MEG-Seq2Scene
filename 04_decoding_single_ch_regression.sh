for sub in $(python configs/config.py)
do
	# for freq_band in low_high low_low high_low high_vlow low_vlow low_vhigh high_vhigh vhigh_vhigh vhigh_high vhigh_low
	# do

	echo "python 04_decoding_single_ch_regression.py -w \
--label Complexity --train-query \"Complexity\" \
-s $sub --train-cond 'two_objects' \
--windows \"2.5,3\" --windows \"2.9,3.1\" --windows \"4.2,4.4\" \
--windows \"3,3.5\" --windows \"4,5\" --windows \"5.5,6.5\" "


done