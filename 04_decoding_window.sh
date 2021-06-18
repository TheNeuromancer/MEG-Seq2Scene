for sub in $(python configs/config.py)
do
	# for freq_band in low_high low_low high_low high_vlow low_vlow low_vhigh high_vhigh vhigh_vhigh vhigh_high vhigh_low
	# do

## OBJECTS
	echo "python 04_decoding_window.py -w \
-s $sub --test-all-times --label AllObjectsfull \
--train-cond 'two_objects' \
--train-query \"Shape1+Colour1\" \
--windows \"0.,1.4\" \
--train-cond 'one_object' \
--train-query \"Shape1+Colour1\" \
--windows \"0.,1.4\" \
--train-cond 'two_objects' \
--train-query \"Shape2+Colour2\" \
--windows \"1.8,3.2\" "



##### COLORS #####
	## FULL
	echo "python 04_decoding_window.py -w \
-s $sub --test-all-times --label AllCfull \
--train-cond 'two_objects' \
--train-query \"Colour1\" \
--windows \"0.6,1.4\" \
--train-cond 'one_object' \
--train-query \"Colour1\" \
--windows \"0.6,1.4\" \
--train-cond 'two_objects' \
--train-query \"Colour2\" \
--windows \"2.4,3.2\" "

	## EARLY
	echo "python 04_decoding_window.py -w \
-s $sub --test-all-times --label AllCearly \
--train-cond 'two_objects' \
--train-query \"Colour1\" \
--windows \"0.7,1.\" \
--train-cond 'one_object' \
--train-query \"Colour1\" \
--windows \"0.7,1.\" \
--train-cond 'two_objects' \
--train-query \"Colour2\" \
--windows \"2.5, 2.79\" "

	## LATE
	echo "python 04_decoding_window.py -w \
-s $sub --test-all-times --label AllClate \
--train-cond 'two_objects' \
--train-query \"Colour1\" \
--windows \"1.05,1.35\" \
--train-cond 'one_object' \
--train-query \"Colour1\" \
--windows \"1.05,1.35\" \
--train-cond 'two_objects' \
--train-query \"Colour2\" \
--windows \"2.85, 3.15\" "


##### SHAPES #####
	## FULL
	echo "python 04_decoding_window.py -w \
-s $sub --test-all-times --label AllSfull \
--train-cond 'two_objects' \
--train-query \"Shape1\" \
--windows \"0.,.8\" \
--train-cond 'one_object' \
--train-query \"Shape1\" \
--windows \"0.,.8\" \
--train-cond 'two_objects' \
--train-query \"Shape2\" \
--windows \"1.8,2.6\" "

	## EARLY
	echo "python 04_decoding_window.py -w \
-s $sub --test-all-times --label AllSearly \
--train-cond 'two_objects' \
--train-query \"Shape1\" \
--windows \".1,.4\" \
--train-cond 'one_object' \
--train-query \"Shape1\" \
--windows \".1,.4\" \
--train-cond 'two_objects' \
--train-query \"Shape2\" \
--windows \"1.9,2.2\" "

## LATE
	echo "python 04_decoding_window.py -w \
-s $sub --test-all-times --label AllSlate \
--train-cond 'two_objects' \
--train-query \"Shape1\" \
--windows \".45,.75\" \
--train-cond 'one_object' \
--train-query \"Shape1\" \
--windows \".45,.75\" \
--train-cond 'two_objects' \
--train-query \"Shape2\" \
--windows \"2.25,2.55\" "

done
