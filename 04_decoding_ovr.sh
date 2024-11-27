for sub in $(python configs/config.py)
do
# 	for freq_band in delta theta alpha beta low_gamma high_gamma
# #low_high low_low high_low high_vlow low_vlow low_vhigh high_vhigh vhigh_vhigh vhigh_high vhigh_low
# 	do

# -c v34 ?

# # Train on the localizer on both image and words -- may be usefull later for replays
# 	echo "python 04_decoding_ovr.py -w --timegen -s $sub \
# 	--train-cond 'localizer' --label LocCrossShapes \
# --train-query 'Loc_crossShape' \
# --test-cond 'one_object' \
# --test-query 'Shape1' \
# --test-cond 'two_objects' \
# --test-query \"Shape1\" \
# --test-cond 'two_objects' \
# --test-query \"Shape2\" "

# 	echo "python 04_decoding_ovr.py -w --timegen -s $sub \
# 	--train-cond 'localizer' --label LocCrossColours \
# --train-query 'Loc_crossColour' \
# --test-cond 'one_object' \
# --test-query 'Colour1' \
# --test-cond 'two_objects' \
# --test-query \"Colour1\" \
# --test-cond 'two_objects' \
# --test-query \"Colour2\" "


# # IMG LOCALIZER  TRAIN ON IMAGES AND TEST ON WORDS
## this does not work. Train on all images and test on their corresponding word.
	echo "python 04_decoding_ovr.py -w --timegen -s $sub \
	--train-cond 'localizer' --label Img2WordAll \
--train-query 'Loc_image' \
--test-cond 'localizer' \
--test-query 'Loc_word' "

echo "python 04_decoding_ovr.py -w --timegen -s $sub \
--train-cond 'localizer' --label Img2WordC \
--train-query 'Loc_image_colour' \
--test-cond 'localizer' \
--test-query 'Loc_colour' \
--test-cond 'one_object' \
--test-query 'Colour1' \
--test-cond 'two_objects' \
--test-query 'Colour1' \
--test-cond 'two_objects' \
--test-query 'Colour2' "

echo "python 04_decoding_ovr.py -w --timegen -s $sub \
--train-cond 'localizer' --label Img2WordS \
--train-query 'Loc_image_shape' \
--test-cond 'localizer' \
--test-query 'Loc_shape' \
--test-cond 'one_object' \
--test-query 'Shape1' \
--test-cond 'two_objects' \
--test-query 'Shape1' \
--test-cond 'two_objects' \
--test-query 'Shape2' "


# # LOCALIZER TRAIN ON WORDS AND TEST ON IMAGES
	echo "python 04_decoding_ovr.py -w --timegen -s $sub \
	--train-cond 'localizer' --label Word2ImgAll \
--train-query 'Loc_word' \
--test-cond 'localizer' \
--test-query 'Loc_image' "

echo "python 04_decoding_ovr.py -w --timegen -s $sub \
--train-cond 'localizer' --label Word2ImgC \
--train-query 'Loc_colour' \
--test-cond 'localizer' \
--test-query 'Loc_image_colour' \
--test-cond 'one_object' \
--test-query 'Colour1' \
--test-cond 'two_objects' \
--test-query 'Colour1' \
--test-cond 'two_objects' \
--test-query 'Colour2' "

echo "python 04_decoding_ovr.py -w --timegen -s $sub \
--train-cond 'localizer' --label Word2ImgS \
--train-query 'Loc_shape' \
--test-cond 'localizer' \
--test-query 'Loc_image_shape' \
--test-cond 'one_object' \
--test-query 'Shape1' \
--test-cond 'two_objects' \
--test-query 'Shape1' \
--test-cond 'two_objects' \
--test-query 'Shape2' "



## COLORS
# 	# train words only
	echo "python 04_decoding_ovr.py -w --train-cond 'localizer' --label Colour \
--timegen -s $sub \
--train-query \"Loc_colour\" \
--test-cond 'one_object' \
--test-query \"Colour1\" \
--test-cond 'two_objects' \
--test-query \"Colour1\" \
--test-cond 'two_objects' \
--test-query \"Colour2\" "
# # --split-queries \"Matching=='match'\" \
# # --split-queries \"Matching=='nonmatch'\" \

	# train on one object
	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label Colour \
--train-cond 'one_object' \
--train-query \"Colour1\" \
--test-cond 'localizer' \
--test-query \"Loc_colour\" \
--test-cond 'localizer' \
--test-query \"Loc_image_colour\" 
--test-cond 'two_objects' \
--test-query \"Colour1\" \
--test-cond 'two_objects' \
--test-query \"Colour2\" "
# --split-queries \"Matching=='match'\" \
# --split-queries \"Change.str.contains('colour')\" "
# --split-queries \"Matching=='match'\" "
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \

	# train on two objects
	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label Colour1 \
--train-cond 'two_objects' \
--train-query \"Colour1\" \
--test-cond 'one_object' \
--test-query \"Colour1\" \
--test-cond 'two_objects' \
--test-query \"Colour2\" "
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \

	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label Colour2 \
--train-cond 'two_objects' \
--train-query \"Colour2\" \
--test-cond 'two_objects' \
--test-query \"Colour1\" \
--test-cond 'one_object' \
--test-query \"Colour1\" "
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \

## SHAPE
# 	# train owords only
# 	echo "python 04_decoding_ovr.py -w --train-cond 'localizer' --label Shape \
# --timegen -s $sub \
# --train-query \"Loc_shape\" \
# --test-cond 'one_object' \
# --test-query \"Shape1\" \
# --test-cond 'two_objects' \
# --test-query \"Shape1\" \
# --test-cond 'two_objects' \
# --test-query \"Shape2\" "
# # --split-queries \"Matching=='match'\" \
# # --split-queries \"Matching=='nonmatch'\" \

	# train on one object
	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label Shape \
--train-cond 'one_object' \
--train-query \"Shape1\" \
--test-cond 'localizer' \
--test-query \"Loc_shape\" \
--test-cond 'localizer' \
--test-query \"Loc_image_shape\" 
--test-cond 'two_objects' \
--test-query \"Shape1\" \
--test-cond 'two_objects' \
--test-query \"Shape2\" "
# --split-queries \"Matching=='match'\" \
# --split-queries \"Change.str.contains('shape')\" "
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \

	# train on two objects
	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label Shape1 \
--train-cond 'two_objects' \
--train-query \"Shape1\" \
--test-cond 'one_object' \
--test-query \"Shape1\" \
--test-cond 'two_objects' \
--test-query \"Shape2\" "
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \

	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label Shape2 \
--train-cond 'two_objects' \
--train-query \"Shape2\" \
--test-cond 'one_object' \
--test-query \"Shape1\" \
--test-cond 'two_objects' \
--test-query \"Shape1\" "
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \

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
# # --split-queries \"Matching=='match'\" \
# # --split-queries \"Matching=='nonmatch'\" \

# 		# train on scenes 1st obj
# 	echo "python 04_decoding_ovr.py -w \
# --timegen -s $sub --train-cond 'two_objects' --label All1stObj \
# --train-query \"Shape1+Colour1\" \
# --test-cond 'one_object' \
# --test-query \"Shape1+Colour1\" \
# --test-cond 'two_objects' \
# --test-query \"Shape2+Colour2\" "
# # --split-queries \"Matching=='match'\" \
# # --split-queries \"Matching=='nonmatch'\" \

# 		# train on scenes 2nd obj
# 	echo "python 04_decoding_ovr.py -w \
# --timegen -s $sub --train-cond 'two_objects' --label All2ndObj \
# --train-query \"Shape2+Colour2\" \
# --test-cond 'one_object' \
# --test-query \"Shape1+Colour1\" \
# --test-cond 'two_objects' \
# --test-query \"Shape1+Colour1\" "
# # --split-queries \"Matching=='match'\" \
# # --split-queries \"Matching=='nonmatch'\" \


## COMPLEXITY
# 	echo "python 04_decoding_ovr.py -w --timegen -s $sub \
# --train-cond 'two_objects' --label SameShape \
# --train-query \"SameShape\" \
# --test-cond 'two_objects' \
# --test-query \"SameColour\" "
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \

# echo "python 04_decoding_ovr.py -w --timegen -s $sub \
# --train-cond 'two_objects' --label SameColour \
# --train-query \"SameColour\" \
# --test-cond 'two_objects' \
# --test-query \"SameShape\" "
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \

# echo "python 04_decoding_ovr.py -w --timegen -s $sub \
# --train-cond 'two_objects' --label SameObject \
# --train-query \"SameObject\"  "
# # --split-queries \"Matching=='match'\" \
# # --split-queries \"Matching=='nonmatch'\" \


## TRAIN OBJECTS ON SCENES (ORDER OF THE VISUAL SCENE)

# 		# RIGHT OBJECT
# 	echo "python 04_decoding_ovr.py -w \
# --label RightColour --timegen -s $sub \
# --train-cond 'two_objects' \
# --train-query \"Right_color\" "

# echo "python 04_decoding_ovr.py -w \
# --label RightShape --timegen -s $sub \
# --train-cond 'two_objects' \
# --train-query \"Right_shape\" "

# echo "python 04_decoding_ovr.py -w \
# --label RightNotLColour --timegen -s $sub \
# --train-cond 'two_objects' \
# --train-query \"RightNotL_color\" "

# echo "python 04_decoding_ovr.py -w \
# --label RightNotLShape --timegen -s $sub \
# --train-cond 'two_objects' \
# --train-query \"RightNotL_shape\" "

# echo "python 04_decoding_ovr.py -w \
# --label AllRightObject --timegen -s $sub \
# --train-cond 'two_objects' \
# --train-query \"Right_obj\" "
# # --split-queries \"Matching=='match'\" \
# # --split-queries \"Matching=='nonmatch'\" \
# # --split-queries \"Flash==0\" \
# # --split-queries \"Flash==1\" \
# # --test-cond 'one_object' \
# # --test-query \"Shape1+Colour1\" "
	
# 		# LEFT OBJECT
# 	echo "python 04_decoding_ovr.py -w \
# --label LeftColour --timegen -s $sub \
# --train-cond 'two_objects' \
# --train-query \"Left_color\" "

# echo "python 04_decoding_ovr.py -w \
# --label LeftShape --timegen -s $sub \
# --train-cond 'two_objects' \
# --train-query \"Left_shape\" "


# 	echo "python 04_decoding_ovr.py -w \
# --label LeftNotR_olour --timegen -s $sub \
# --train-cond 'two_objects' \
# --train-query \"LeftNotR_color\" "

# echo "python 04_decoding_ovr.py -w \
# --label LeftNotR_hape --timegen -s $sub \
# --train-cond 'two_objects' \
# --train-query \"LeftNotR_shape\" "

# 	echo "python 04_decoding_ovr.py -w \
# --label AllLeftObject --timegen -s $sub \
# --train-cond 'two_objects' \
# --train-query \"Left_obj\" "
# # --split-queries \"Matching=='match'\" \
# # --split-queries \"Matching=='nonmatch'\" \
# # --split-queries \"Flash==0\" \
# # --split-queries \"Flash==1\" \
# # --test-cond 'one_object' \
# # --test-query \"Shape1+Colour1\" "


## RELATION
	echo "python 04_decoding_ovr.py -w \
--timegen -s $sub --label Relation \
--train-cond 'two_objects' \
--train-query \"Relation\" "
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \

# ## MISMATCH SIDE
# 	echo "python 04_decoding_ovr.py -w \
# --timegen -s $sub --label winMismatchSide \
# --train-cond 'two_objects' \
# --train-query \"MismatchSide\" \
# --windows '5,8.5'"

# echo "python 04_decoding_ovr.py -w \
# --timegen -s $sub --label MismatchLeft \
# --train-cond 'two_objects' \
# --train-query \"MismatchLeft\" "

# echo "python 04_decoding_ovr.py -w \
# --timegen -s $sub --label MismatchRight \
# --train-cond 'two_objects' \
# --train-query \"MismatchRight\" "



# # ## BUTTON
# 	echo "python 04_decoding_ovr.py -w \
# --timegen -s $sub --label Button \
# --train-cond 'two_objects' \
# --train-query \"Button\" "
# # --split-queries \"Matching=='match'\" \
# # --split-queries \"Matching=='nonmatch'\" \

# 	echo "python 04_decoding_ovr.py -w \
# --timegen -s $sub --label Button \
# --train-cond 'one_object' \
# --train-query \"Button\" "
# # --split-queries \"Matching=='match'\" \
# # --split-queries \"Matching=='nonmatch'\" \

# 	echo "python 04_decoding_ovr.py -w \
# --timegen -s $sub --label RespButton --response_lock \
# --train-cond 'two_objects' \
# --train-query \"Button\" "
# # --split-queries \"Matching=='match'\" \
# # --split-queries \"Matching=='nonmatch'\" \

# 	echo "python 04_decoding_ovr.py -w \
# --timegen -s $sub --label RespButton --response_lock \
# --train-cond 'one_object' \
# --train-query \"Button\" "
# # --split-queries \"Matching=='match'\" \
# # --split-queries \"Matching=='nonmatch'\" \


# ## FLASH
# 	echo "python 04_decoding_ovr.py -w \
# --timegen -s $sub --label Flash \
# --train-cond 'two_objects' \
# --train-query \"Flash\" "
# # --split-queries \"Matching=='match'\" \
# # --split-queries \"Matching=='nonmatch'\" \


# ## PERF
# 	echo "python 04_decoding_ovr.py -w \
# --timegen -s $sub --label Perf \
# --train-cond 'two_objects' \
# --train-query \"Perf\" \
# --filter ''" # empty filter to overwrite the config with perf filtering
# # --split-queries \"Matching=='match'\" \
# # --split-queries \"Matching=='nonmatch'\" \

# 	echo "python 04_decoding_ovr.py -w \
# --timegen -s $sub --label Perf \
# --train-cond 'one_object' \
# --train-query \"Perf\" \
# --filter ''" # empty filter to overwrite the config with perf filtering
# # --split-queries \"Matching=='match'\" \
# # --split-queries \"Matching=='nonmatch'\" \

# 	echo "python 04_decoding_ovr.py -w \
# --timegen -s $sub --label RespPerf --response_lock \
# --train-cond 'two_objects' \
# --train-query \"Perf\" \
# --filter ''" # empty filter to overwrite the config with perf filtering
# # --split-queries \"Matching=='match'\" \
# # --split-queries \"Matching=='nonmatch'\" \

# 	echo "python 04_decoding_ovr.py -w \
# --timegen -s $sub --label RespPerf --response_lock \
# --train-cond 'one_object' \
# --train-query \"Perf\" \
# --filter ''" # empty filter to overwrite the config with perf filtering
# # --split-queries \"Matching=='match'\" \
# # --split-queries \"Matching=='nonmatch'\" \


# ## MATCHING
# 	echo "python 04_decoding_ovr.py -w \
# --timegen -s $sub --label Matching \
# --train-cond 'two_objects' \
# --train-query \"Matching\" "

# 	echo "python 04_decoding_ovr.py -w \
# --timegen -s $sub --label Matching \
# --train-cond 'one_object' \
# --train-query \"Matching\" "

# 	echo "python 04_decoding_ovr.py -w \
# --timegen -s $sub --label RespMatching --response_lock \
# --train-cond 'two_objects' \
# --train-query \"Matching\" "

# 	echo "python 04_decoding_ovr.py -w \
# --timegen -s $sub --label RespMatching --response_lock \
# --train-cond 'one_object' \
# --train-query \"Matching\" "
# # 		## MISMATCHES

# # 	## ONE OBJECT MISMATCHES
# 	echo "python 04_decoding_ovr.py -w \
# --timegen -s $sub --label ColourMismatch \
# --train-cond 'one_object' \
# --train-query \"ColourMismatch\" "
# # --split-queries \"Flash==0\" \
# # --split-queries \"Flash==1\" \

# 	echo "python 04_decoding_ovr.py -w \
# --timegen -s $sub --label ShapeMismatch \
# --train-cond 'one_object' \
# --train-query \"ShapeMismatch\" "
# # --split-queries \"Flash==0\" \
# # --split-queries \"Flash==1\" \

# 	echo "python 04_decoding_ovr.py -w \
# --timegen -s $sub --label RespColourMismatch --response_lock \
# --train-cond 'one_object' \
# --train-query \"ColourMismatch\" "
# # --split-queries \"Flash==0\" \
# # --split-queries \"Flash==1\" \

# 	echo "python 04_decoding_ovr.py -w \
# --timegen -s $sub --label RespShapeMismatch --response_lock \
# --train-cond 'one_object' \
# --train-query \"ShapeMismatch\" "
# # --split-queries \"Flash==0\" \
# # --split-queries \"Flash==1\" \

# 	## TWO OBJECTS MISMATCHES
# 	echo "python 04_decoding_ovr.py -w -w \
# --timegen -s $sub --label PropMismatch \
# --train-cond 'two_objects' \
# --train-query \"PropMismatch\" \
# --split-queries \"Complexity==0\" \
# --split-queries \"Complexity==1\" \
# --split-queries \"Complexity==2\" "

# 	echo "python 04_decoding_ovr.py -w -w \
# --timegen -s $sub --label BindMismatch \
# --train-cond 'two_objects' \
# --train-query \"BindMismatch\" \
# --split-queries \"Complexity==0\" \
# --split-queries \"Complexity==1\" \
# --split-queries \"Complexity==2\" "

# 	echo "python 04_decoding_ovr.py -w -w \
# --timegen -s $sub --label RelMismatch \
# --train-cond 'two_objects' \
# --train-query \"RelMismatch\" \
# --split-queries \"Complexity==0\" \
# --split-queries \"Complexity==1\" \
# --split-queries \"Complexity==2\" "




# 	echo "python 04_decoding_ovr.py -w -w \
# --timegen -s $sub --label Matching \
# --train-cond 'two_objects' \
# --train-query \"Matching\" \
# --split-queries \"Complexity==0\" \
# --split-queries \"Complexity==1\" \
# --split-queries \"Complexity==2\" \
# --split-queries \"Error_type=='l0'\" \
# --split-queries \"Error_type=='l1'\" \
# --split-queries \"Error_type=='l2'\" "



# 	echo "python 04_decoding_ovr.py -w -w \
# --timegen -s $sub --label Mismatches \
# --train-cond 'two_objects' \
# --train-query \"Mismatches\" \
# --split-queries \"Complexity==0\" \
# --split-queries \"Complexity==1\" \
# --split-queries \"Complexity==2\" \
# --split-queries \"Error_type=='l0'\" \
# --split-queries \"Error_type=='l1'\" \
# --split-queries \"Error_type=='l2'\" "


# # response lock
# 	echo "python 04_decoding_ovr.py -w \
# --timegen -s $sub --label RespPropMismatch --response_lock \
# --train-cond 'two_objects' \
# --train-query \"PropMismatch\" \
# --split-queries \"Complexity==0\" \
# --split-queries \"Complexity==1\" \
# --split-queries \"Complexity==2\" "

# 	echo "python 04_decoding_ovr.py -w \
# --timegen -s $sub --label RespBindMismatch --response_lock \
# --train-cond 'two_objects' \
# --train-query \"BindMismatch\" \
# --split-queries \"Complexity==0\" \
# --split-queries \"Complexity==1\" \
# --split-queries \"Complexity==2\" "

# 	echo "python 04_decoding_ovr.py -w \
# --timegen -s $sub --label RespRelMismatch --response_lock \
# --train-cond 'two_objects' \
# --train-query \"RelMismatch\" \
# --split-queries \"Complexity==0\" \
# --split-queries \"Complexity==1\" \
# --split-queries \"Complexity==2\" "


# 	echo "python 04_decoding_ovr.py -w \
# --timegen -s $sub --label RespMismatches --response_lock \
# --train-cond 'two_objects' \
# --train-query \"Mismatches\" \
# --split-queries \"Complexity==0\" \
# --split-queries \"Complexity==1\" \
# --split-queries \"Complexity==2\" \
# --split-queries \"Error_type=='l0'\" \
# --split-queries \"Error_type=='l1'\" \
# --split-queries \"Error_type=='l2'\" "



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

	# done
done