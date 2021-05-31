for sub in $(python configs/config.py)
do
	# for freq_band in low_high low_low high_low high_vlow low_vlow low_vhigh high_vhigh vhigh_vhigh vhigh_high vhigh_low
	# do

## OBJECTS
	echo "python 04_decoding_window.py -w \ -s $sub --label AllObjects \
--train-cond 'two_objects' \
--train-query \"Shape1+Colour1\" \
--windows \"0., 1.8\" \
--train-cond 'one_object' \
--train-query \"Shape1+Colour1\" \
--windows \"0., 1.8\" \
--train-cond 'two_objects' \
--train-query \"Shape2+Colour2\" \
--windows \"1.8, 3.6\" "




## COLORS
	echo "python 04_decoding_window.py -w \ -s $sub --label AllColors \
--train-cond 'two_objects' \
--train-query \"Colour1\" \
--windows \"0.6, 1.4\" \
--train-cond 'one_object' \
--train-query \"Colour1\" \
--windows \"0.6, 1.4\" \
--train-cond 'two_objects' \
--train-query \"Colour2\" \
--windows \"2.4, 3.2\" "



## SHAPES
	echo "python 04_decoding_window.py -w \ -s $sub --label AllShapes \
--train-cond 'two_objects' \
--train-query \"Shape1\" \
--windows \"0., .8\" \
--train-cond 'one_object' \
--train-query \"Shape1\" \
--windows \"0., .8\" \
--train-cond 'two_objects' \
--train-query \"Shape2\" \
--windows \"1.8, 2.6\" "



done