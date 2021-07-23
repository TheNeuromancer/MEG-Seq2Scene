for sub in $(python configs/config.py)
do
	# for freq_band in low_high low_low high_low high_vlow low_vlow low_vhigh high_vhigh vhigh_vhigh vhigh_high vhigh_low
	# do


# FOR NOW ONLY INDIV OBJECTS DECODING
## COLORS
## Not the same query ...
# 	# train on localizer words only
# 	echo "python 04_decoding_ovr.py -w --train-cond 'localizer' --label Colour \
# --timegen -s $sub \
# --train-query \"Loc_word\" \
# --test-cond 'one_object' \
# --test-query-1 \"Colour1\" \
# --test-cond 'two_objects' "
# --train-query \"Loc_word in ['${colours[0]}', 'img_${colours[0]}']\" \

	# train on one object
	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label Colour \
--train-cond 'one_object' \
--train-query \"Colour1\" \
--test-cond 'two_objects' \
--test-query \"Colour1\" \
--test-cond 'two_objects' \
--test-query \"Colour2\" "

	# train on two objects
	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label Colour1 \
--train-cond 'two_objects' \
--train-query \"Colour1\" \
--test-cond 'one_object' \
--test-query \"Colour1\" \
--test-cond 'two_objects' \
--test-query \"Colour2\" "

	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label Colour2 \
--train-cond 'two_objects' \
--train-query \"Colour2\" \
--test-cond 'two_objects' \
--test-query \"Colour1\" \
--test-cond 'one_object' \
--test-query \"Colour1\" "

## SHAPE
## Not the same query ...
# 	# train on localizer words only
# 	echo "python 04_decoding_ovr.py -w --train-cond 'localizer' --label Shape \
# --timegen -s $sub \
# --train-query \"Loc_word\" \
# --test-cond 'one_object' \
# --test-cond 'two_objects' "

	# train on one object
	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label Shape \
--train-cond 'one_object' \
--train-query \"Shape1\" \
--test-cond 'two_objects' \
--test-query \"Shape1\" \
--test-cond 'two_objects' \
--test-query \"Shape2\" "

	# train on two objects
	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label Shape1 \
--train-cond 'two_objects' \
--train-query \"Shape1\" \
--test-cond 'one_object' \
--test-query \"Shape1\" \
--test-cond 'two_objects' \
--test-query \"Shape2\" "

	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label Shape2 \
--train-cond 'two_objects' \
--train-query \"Shape2\" \
--test-cond 'one_object' \
--test-query \"Shape1\" \
--test-cond 'two_objects' \
--test-query \"Shape1\" "


## OBJECTS
		# train on all other trials, gen to first, then to 2nd object
	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --train-cond 'one_object' --label AllObject \
--train-query \"Shape1+Colour1\" \
--test-cond 'two_objects' \
--test-query \"Shape1+Colour1\" \
--test-cond 'two_objects' \
--test-query \"Shape2+Colour2\" \
--test-cond 'two_objects' \
--test-query \"Right_obj\" \
--test-cond 'two_objects' \
--test-query \"Left_obj\" "

		# train on scenes 1st obj
	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --train-cond 'two_objects' --label All1stObj \
--train-query \"Shape1+Colour1\" \
--test-cond 'one_object' \
--test-query \"Shape1+Colour1\" \
--test-cond 'two_objects' \
--test-query \"Shape2+Colour2\" "

		# train on scenes 2nd obj
	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --train-cond 'two_objects' --label All2ndObj \
--train-query \"Shape2+Colour2\" \
--test-cond 'one_object' \
--test-query \"Shape1+Colour1\" \
--test-cond 'two_objects' \
--test-query \"Shape1+Colour1\" "


## TRAIN OBJECTS ON SCENES (ORDER OF THE VISUAL SCENE)

		# RIGHT OBJECT
	echo "python 04_decoding_ovr.py -w \
--label AllRightObject --timegen -s $sub \
--train-cond 'two_objects' \
--train-query \"Right_obj\" \
--test-cond 'one_object' \
--test-query \"Shape1+Colour1\" "
	
		# LEFT OBJECT
	echo "python 04_decoding_ovr.py -w \
--label AllLeftObject --timegen -s $sub \
--train-cond 'two_objects' \
--train-query \"Left_obj\" \
--test-cond 'one_object' \
--test-query \"Shape1+Colour1\" "


## RELATION
	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label Relation \
--train-cond 'two_objects' \
--train-query \"Relation\" "



## USE WINDOWS AROUND EACH COND

	## COLOUR
	# train on one object
	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label winColour \
--train-cond 'one_object' \
--windows \"0.6,1.4\" \
--train-query \"Colour1\" \
--test-cond 'two_objects' \
--windows \"0.6,1.4\" \
--test-query \"Colour1\" \
--test-cond 'two_objects' \
--windows \"2.4, 3.2\" \
--test-query \"Colour2\" "

	# train on two objects
	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label winColour1 \
--train-cond 'two_objects' \
--windows \"0.6,1.4\" \
--train-query \"Colour1\" \
--test-cond 'one_object' \
--windows \"0.6,1.4\" \
--test-query \"Colour1\" \
--test-cond 'two_objects' \
--windows \"2.4, 3.2\" \
--test-query \"Colour2\" "

	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label winColour2 \
--train-cond 'two_objects' \
--windows \"2.4, 3.2\" \
--train-query \"Colour2\" \
--test-cond 'two_objects' \
--windows \"0.6,1.4\" \
--test-query \"Colour1\" \
--test-cond 'one_object' \
--windows \"0.6,1.4\" \
--test-query \"Colour1\" "


	## SHAPES
	# train on one object
	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label winShape \
--train-cond 'one_object' \
--windows \"0.,.8\" \
--train-query \"Shape1\" \
--test-cond 'two_objects' \
--windows \"0.,.8\" \
--test-query \"Shape1\" \
--test-cond 'two_objects' \
--windows \"1.8,2.6\" \
--test-query \"Shape2\" "

	# train on two objects
	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label winShape1 \
--train-cond 'two_objects' \
--windows \"0.,.8\" \
--train-query \"Shape1\" \
--test-cond 'one_object' \
--windows \"0.,.8\" \
--test-query \"Shape1\" \
--test-cond 'two_objects' \
--windows \"1.8,2.6\" \
--test-query \"Shape2\" "

	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label winShape2 \
--train-cond 'two_objects' \
--windows \"1.8,2.6\" \
--train-query \"Shape2\" \
--test-cond 'one_object' \
--windows \"0.,.8\" \
--test-query \"Shape1\" \
--test-cond 'two_objects' \
--windows \"0.,.8\" \
--test-query \"Shape1\" "


	## OBJECTS
	# train on all other trials, gen to first, then to 2nd object
	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label winAllObject \
--train-cond 'one_object' \
--windows \"0.,1.4\" \
--train-query \"Shape1+Colour1\" \
--test-cond 'two_objects' \
--test-query \"Shape1+Colour1\" \
--windows \"0.,1.4\" \
--test-cond 'two_objects' \
--windows \"1.8,3.2\" \
--test-query \"Shape2+Colour2\" "

	# train on scenes 1st obj
	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label winAll1stObj \
--train-cond 'two_objects' \
--windows \"0.,1.4\" \
--train-query \"Shape1+Colour1\" \
--test-cond 'one_object' \
--windows \"0.,1.4\" \
--test-query \"Shape1+Colour1\" \
--test-cond 'two_objects' \
--windows \"1.8,3.2\" \
--test-query \"Shape2+Colour2\" "

	# train on scenes 2nd obj
	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label winAll2ndObj \
--split-queries \"Matching=='match'\" \
--split-queries \"Matching=='nonmatch'\" \
--train-cond 'two_objects' \
--windows \"1.8,3.2\" \
--train-query \"Shape2+Colour2\" \
--test-cond 'one_object' \
--windows \"0.,1.4\" \
--test-query \"Shape1+Colour1\" \
--test-cond 'two_objects' \
--windows \"0.,1.4\" \
--test-query \"Shape1+Colour1\" "
done