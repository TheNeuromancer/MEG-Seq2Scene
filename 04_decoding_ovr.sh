for sub in $(python configs/config.py)
do
	# for freq_band in low_high low_low high_low high_vlow low_vlow low_vhigh high_vhigh vhigh_vhigh vhigh_high vhigh_low
	# do


# FOR NOW ONLY INDIV OBJECTS DECODING
## COLORS
# 	# train on localizer words only
# 	echo "python 04_decoding_ovr.py -w --train-cond 'localizer' --label Colour \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --timegen -s $sub \
# --train-query \"Loc_word\" \
# --test-cond 'one_object' \
# --test-query \"Colour1\" \
# --test-cond 'two_objects' \
# --test-query \"Colour1\" "

	# train on one object
	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label Colour \
--split-queries \"Matching=='match'\" \
--split-queries \"Matching=='nonmatch'\" \
--train-cond 'one_object' \
--train-query \"Colour1\" \
--test-cond 'two_objects' \
--test-query \"Colour1\" \
--test-cond 'two_objects' \
--test-query \"Colour2\" "

	# train on two objects
	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label Colour1 \
--split-queries \"Matching=='match'\" \
--split-queries \"Matching=='nonmatch'\" \
--train-cond 'two_objects' \
--train-query \"Colour1\" \
--test-cond 'one_object' \
--test-query \"Colour1\" \
--test-cond 'two_objects' \
--test-query \"Colour2\" "

	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label Colour2 \
--split-queries \"Matching=='match'\" \
--split-queries \"Matching=='nonmatch'\" \
--train-cond 'two_objects' \
--train-query \"Colour2\" \
--test-cond 'two_objects' \
--test-query \"Colour1\" \
--test-cond 'one_object' \
--test-query \"Colour1\" "

## SHAPE
# 	# train on localizer words only
# 	echo "python 04_decoding_ovr.py -w --train-cond 'localizer' --label Shape \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --timegen -s $sub \
# --train-query \"Loc_word\" \
# --test-cond 'one_object' \
# --test-query \"Shape1\" \
# --test-cond 'two_objects' \
# --test-query \"Shape1\" "

	# train on one object
	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label Shape \
--split-queries \"Matching=='match'\" \
--split-queries \"Matching=='nonmatch'\" \
--train-cond 'one_object' \
--train-query \"Shape1\" \
--test-cond 'two_objects' \
--test-query \"Shape1\" \
--test-cond 'two_objects' \
--test-query \"Shape2\" "

	# train on two objects
	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label Shape1 \
--split-queries \"Matching=='match'\" \
--split-queries \"Matching=='nonmatch'\" \
--train-cond 'two_objects' \
--train-query \"Shape1\" \
--test-cond 'one_object' \
--test-query \"Shape1\" \
--test-cond 'two_objects' \
--test-query \"Shape2\" "

	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label Shape2 \
--split-queries \"Matching=='match'\" \
--split-queries \"Matching=='nonmatch'\" \
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
--split-queries \"Matching=='match'\" \
--split-queries \"Matching=='nonmatch'\" \
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
--split-queries \"Matching=='match'\" \
--split-queries \"Matching=='nonmatch'\" \
--train-query \"Shape1+Colour1\" \
--test-cond 'one_object' \
--test-query \"Shape1+Colour1\" \
--test-cond 'two_objects' \
--test-query \"Shape2+Colour2\" "

		# train on scenes 2nd obj
	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --train-cond 'two_objects' --label All2ndObj \
--split-queries \"Matching=='match'\" \
--split-queries \"Matching=='nonmatch'\" \
--train-query \"Shape2+Colour2\" \
--test-cond 'one_object' \
--test-query \"Shape1+Colour1\" \
--test-cond 'two_objects' \
--test-query \"Shape1+Colour1\" "


## COMPLEXITY
	echo "python 04_decoding_ovr.py -w --timegen -s $sub \
--train-cond 'two_objects' --label SameShape \
--split-queries \"Matching=='match'\" \
--split-queries \"Matching=='nonmatch'\" \
--train-query \"SameShape\" \
--test-cond 'two_objects' \
--test-query \"SameColour\" "

echo "python 04_decoding_ovr.py -w --timegen -s $sub \
--train-cond 'two_objects' --label SameColour \
--split-queries \"Matching=='match'\" \
--split-queries \"Matching=='nonmatch'\" \
--train-query \"SameColour\" \
--test-cond 'two_objects' \
--test-query \"SameShape\" "

echo "python 04_decoding_ovr.py -w --timegen -s $sub \
--train-cond 'two_objects' --label SameObject \
--split-queries \"Matching=='match'\" \
--split-queries \"Matching=='nonmatch'\" \
--train-query \"SameObject\"  "



## TRAIN OBJECTS ON SCENES (ORDER OF THE VISUAL SCENE)

# 		# RIGHT OBJECT
# 	echo "python 04_decoding_ovr.py -w \
# --label AllRightObject --timegen -s $sub \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --split-queries \"Flash==0\" \
# --split-queries \"Flash==1\" \
# --train-cond 'two_objects' \
# --train-query \"Right_obj\" \
# --test-cond 'one_object' \
# --test-query \"Shape1+Colour1\" "
	
# 		# LEFT OBJECT
# 	echo "python 04_decoding_ovr.py -w \
# --label AllLeftObject --timegen -s $sub \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --split-queries \"Flash==0\" \
# --split-queries \"Flash==1\" \
# --train-cond 'two_objects' \
# --train-query \"Left_obj\" \
# --test-cond 'one_object' \
# --test-query \"Shape1+Colour1\" "


## RELATION
	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label Relation \
--split-queries \"Matching=='match'\" \
--split-queries \"Matching=='nonmatch'\" \
--train-cond 'two_objects' \
--train-query \"Relation\" "


## BUTTON
	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label Button \
--split-queries \"Matching=='match'\" \
--split-queries \"Matching=='nonmatch'\" \
--train-cond 'two_objects' \
--train-query \"Button\" "

	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label Button \
--split-queries \"Matching=='match'\" \
--split-queries \"Matching=='nonmatch'\" \
--train-cond 'one_object' \
--train-query \"Button\" "

## FLASH
	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label Flash \
--split-queries \"Matching=='match'\" \
--split-queries \"Matching=='nonmatch'\" \
--train-cond 'two_objects' \
--train-query \"Flash\" "


## PERF
	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label Perf \
--split-queries \"Matching=='match'\" \
--split-queries \"Matching=='nonmatch'\" \
--train-cond 'two_objects' \
--train-query \"Perf\" \
--filter ''" # empty filter to overwrite the config with perf filtering

	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label Perf \
--split-queries \"Matching=='match'\" \
--split-queries \"Matching=='nonmatch'\" \
--train-cond 'one_object' \
--train-query \"Perf\" \
--filter ''" # empty filter to overwrite the config with perf filtering

## MATCHING
	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label Matching \
--train-cond 'two_objects' \
--train-query \"Matching\" "

	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label Matching \
--train-cond 'one_object' \
--train-query \"Matching\" "

		## MISMATCHES

	## ONE OBJECT MISMATCHES
	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label ColourMismatch \
--split-queries \"Flash==0\" \
--split-queries \"Flash==1\" \
--train-cond 'one_object' \
--train-query \"ColourMismatch\" "

	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label ShapeMismatch \
--split-queries \"Flash==0\" \
--split-queries \"Flash==1\" \
--train-cond 'one_object' \
--train-query \"ShapeMismatch\" "


	## TWO OBJECTS MISMATCHES
	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label PropMismatch \
--train-cond 'two_objects' \
--train-query \"PropMismatch\" "

	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label BindMismatch \
--train-cond 'two_objects' \
--train-query \"BindMismatch\" "

	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label RelMismatch \
--train-cond 'two_objects' \
--train-query \"RelMismatch\" "


	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label Mismatches \
--train-cond 'two_objects' \
--train-query \"Mismatches\" "



## USE WINDOWS AROUND EACH COND

# 	## COLOUR
# 	# train on one object
# 	echo "python 04_decoding_ovr.py -w \
# --timegen -s $sub --label winColour \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --train-cond 'one_object' \
# --windows \"0.6,1.4\" \
# --train-query \"Colour1\" \
# --test-cond 'two_objects' \
# --windows \"0.6,1.4\" \
# --test-query \"Colour1\" \
# --test-cond 'two_objects' \
# --windows \"2.4, 3.2\" \
# --test-query \"Colour2\" "

# 	# train on two objects
# 	echo "python 04_decoding_ovr.py -w \
# --timegen -s $sub --label winColour1 \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --train-cond 'two_objects' \
# --windows \"0.6,1.4\" \
# --train-query \"Colour1\" \
# --test-cond 'one_object' \
# --windows \"0.6,1.4\" \
# --test-query \"Colour1\" \
# --test-cond 'two_objects' \
# --windows \"2.4, 3.2\" \
# --test-query \"Colour2\" "

# 	echo "python 04_decoding_ovr.py -w \
# --timegen -s $sub --label winColour2 \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --train-cond 'two_objects' \
# --windows \"2.4, 3.2\" \
# --train-query \"Colour2\" \
# --test-cond 'two_objects' \
# --windows \"0.6,1.4\" \
# --test-query \"Colour1\" \
# --test-cond 'one_object' \
# --windows \"0.6,1.4\" \
# --test-query \"Colour1\" "


# 	## SHAPES
# 	# train on one object
# 	echo "python 04_decoding_ovr.py -w \
# --timegen -s $sub --label winShape \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --train-cond 'one_object' \
# --windows \"0.,.8\" \
# --train-query \"Shape1\" \
# --test-cond 'two_objects' \
# --windows \"0.,.8\" \
# --test-query \"Shape1\" \
# --test-cond 'two_objects' \
# --windows \"1.8,2.6\" \
# --test-query \"Shape2\" "

# 	# train on two objects
# 	echo "python 04_decoding_ovr.py -w \
# --timegen -s $sub --label winShape1 \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --train-cond 'two_objects' \
# --windows \"0.,.8\" \
# --train-query \"Shape1\" \
# --test-cond 'one_object' \
# --windows \"0.,.8\" \
# --test-query \"Shape1\" \
# --test-cond 'two_objects' \
# --windows \"1.8,2.6\" \
# --test-query \"Shape2\" "

# 	echo "python 04_decoding_ovr.py -w \
# --timegen -s $sub --label winShape2 \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --train-cond 'two_objects' \
# --windows \"1.8,2.6\" \
# --train-query \"Shape2\" \
# --test-cond 'one_object' \
# --windows \"0.,.8\" \
# --test-query \"Shape1\" \
# --test-cond 'two_objects' \
# --windows \"0.,.8\" \
# --test-query \"Shape1\" "


# 	## OBJECTS
# 	# train on all other trials, gen to first, then to 2nd object
# 	echo "python 04_decoding_ovr.py -w \
# --timegen -s $sub --label winAllObject \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --train-cond 'one_object' \
# --windows \"0.,1.4\" \
# --train-query \"Shape1+Colour1\" \
# --test-cond 'two_objects' \
# --test-query \"Shape1+Colour1\" \
# --windows \"0.,1.4\" \
# --test-cond 'two_objects' \
# --windows \"1.8,3.2\" \
# --test-query \"Shape2+Colour2\" "

# 	# train on scenes 1st obj
# 	echo "python 04_decoding_ovr.py -w \
# --timegen -s $sub --label winAll1stObj \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --train-cond 'two_objects' \
# --windows \"0.,1.4\" \
# --train-query \"Shape1+Colour1\" \
# --test-cond 'one_object' \
# --windows \"0.,1.4\" \
# --test-query \"Shape1+Colour1\" \
# --test-cond 'two_objects' \
# --windows \"1.8,3.2\" \
# --test-query \"Shape2+Colour2\" "

# 	# train on scenes 2nd obj
# 	echo "python 04_decoding_ovr.py -w \
# --timegen -s $sub --label winAll2ndObj \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --train-cond 'two_objects' \
# --windows \"1.8,3.2\" \
# --train-query \"Shape2+Colour2\" \
# --test-cond 'one_object' \
# --windows \"0.,1.4\" \
# --test-query \"Shape1+Colour1\" \
# --test-cond 'two_objects' \
# --windows \"0.,1.4\" \
# --test-query \"Shape1+Colour1\" "



# ## During the delay
# ## OBJECTS
# 	echo "python 04_decoding_ovr.py -w \
# -s $sub --label winObjectsdelay \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --train-cond 'two_objects' \
# --train-query \"Shape1+Colour1\" \
# --windows \"3.5,5.\" \
# --train-cond 'two_objects' \
# --train-query \"Shape2+Colour2\" \
# --windows \"3.5,5.\" "

# ## SHAPES
# 	echo "python 04_decoding_ovr.py -w \
# -s $sub --label winShapesdelay \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --train-cond 'two_objects' \
# --train-query \"Shape1\" \
# --windows \"3.5,5.\" \
# --train-cond 'two_objects' \
# --train-query \"Shape2\" \
# --windows \"3.5,5.\" "

# ## COLORS
# 	echo "python 04_decoding_ovr.py -w \
# -s $sub --label winColoursdelay \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --train-cond 'two_objects' \
# --train-query \"Colour1\" \
# --windows \"3.5,5.\" \
# --train-cond 'two_objects' \
# --train-query \"Colour2\" \
# --windows \"3.5,5.\" "


# 	echo "python 04_decoding_ovr.py -w \
# -s $sub --label SideObjectsdelay \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --train-cond 'two_objects' \
# --train-query \"Left_obj\" \
# --windows \"3.5,5.\" \
# --train-cond 'two_objects' \
# --train-query \"Right_obj\" \
# --windows \"3.5,5.\" "


done