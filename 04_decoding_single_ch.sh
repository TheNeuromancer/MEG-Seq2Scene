for sub in $(python configs/config.py)
do
	# for freq_band in low_high low_low high_low high_vlow low_vlow low_vhigh high_vhigh vhigh_vhigh vhigh_high vhigh_low
	# do


# FOR NOW ONLY INDIV OBJECTS DECODING
## COLORS
## Not the same query ...
# 	# train on localizer words only
# 	echo "python 04_decoding_single_ch.py -w --train-cond 'localizer' --label Colour \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# -s $sub \
# --train-query \"Loc_word\" \
# --test-cond 'one_object' \
# --test-query-1 \"Colour1\" \
# --test-cond 'two_objects' "
# --train-query \"Loc_word in ['${colours[0]}', 'img_${colours[0]}']\" \

# 	# train on one object
# 	echo "python 04_decoding_single_ch.py -w \
# -s $sub --label Colour \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --train-cond 'one_object' \
# --train-query \"Colour1\" "
# # --test-cond 'two_objects' \
# # --test-query \"Colour1\" \
# # --test-cond 'two_objects' \
# # --test-query \"Colour2\" "

# 	# kind of mismatch one obj
# 	echo "python 04_decoding_single_ch.py -w \
# -s $sub --label ColourMismatch \
# --train-cond 'one_object' \
# --train-query \"ColourMismatch\" "
# 	echo "python 04_decoding_single_ch.py -w \
# -s $sub --label ShapeMismatch \
# --train-cond 'one_object' \
# --train-query \"ShapeMismatch\" "

# 	# train on two objects
# 	echo "python 04_decoding_single_ch.py -w \
# -s $sub --label Colour1 \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --train-cond 'two_objects' \
# --train-query \"Colour1\" \
# --test-cond 'two_objects' \
# --test-query \"Colour2\" "
# # --test-cond 'one_object' \
# # --test-query \"Colour1\" \

# 	echo "python 04_decoding_single_ch.py -w \
# -s $sub --label Colour2 \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --train-cond 'two_objects' \
# --train-query \"Colour2\" \
# --test-cond 'two_objects' \
# --test-query \"Colour1\" "
# # --test-cond 'one_object' \
# # --test-query \"Colour1\" 

# ## SHAPE
# ## Not the same query ...
# 	# train on localizer words only
# 	echo "python 04_decoding_single_ch.py -w --train-cond 'localizer' --label Shape \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# -s $sub \
# --train-query \"Loc_word\" \
# --test-cond 'one_object' \
# --test-cond 'two_objects' "

# 	# train on one object
# 	echo "python 04_decoding_single_ch.py -w \
# -s $sub --label Shape \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --train-cond 'one_object' \
# --train-query \"Shape1\" "
# # --test-cond 'two_objects' \
# # --test-query \"Shape1\" \
# # --test-cond 'two_objects' \
# # --test-query \"Shape2\" "

# 	# train on two objects
# 	echo "python 04_decoding_single_ch.py -w \
# -s $sub --label Shape1 \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --train-cond 'two_objects' \
# --train-query \"Shape1\" \
# --test-cond 'two_objects' \
# --test-query \"Shape2\" "
# # --test-cond 'one_object' \
# # --test-query \"Shape1\" \

# 	echo "python 04_decoding_single_ch.py -w \
# -s $sub --label Shape2 \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --train-cond 'two_objects' \
# --train-query \"Shape2\" \
# --test-cond 'two_objects' \
# --test-query \"Shape1\" "
# # --test-cond 'one_object' \
# # --test-query \"Shape1\" 


## LOCALIZER WORDS
	# train on localizer words only
	echo "python 04_decoding_single_ch.py -w --train-cond 'localizer' \
--label Loc_word -s $sub --c v25_config \
--train-query \"Loc_word\" "

# train on localizer words + images
	echo "python 04_decoding_single_ch.py -w --train-cond 'localizer' \
--label Loc_all -s $sub --c v25_config \
--train-query \"Loc_all\" "


# ## OBJECTS
		# train on all other trials, gen to first, then to 2nd object
	echo "python 04_decoding_single_ch.py -w \
-s $sub --c v25_config --train-cond 'one_object' --label AllObject \
--train-query \"Shape1+Colour1\" "
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --test-cond 'two_objects' \
# --test-query \"Shape1+Colour1\" \
# --test-cond 'two_objects' \
# --test-query \"Shape2+Colour2\" \
# --test-cond 'two_objects' \
# --test-query \"Right_obj\" \
# --test-cond 'two_objects' \
# --test-query \"Left_obj\" "

# 		# train on scenes 1st obj
# 	echo "python 04_decoding_single_ch.py -w \
# -s $sub --train-cond 'two_objects' --label All1stObj \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --train-query \"Shape1+Colour1\" \
# --test-cond 'two_objects' \
# --test-query \"Shape2+Colour2\" "
# # --test-cond 'one_object' \
# # --test-query \"Shape1+Colour1\" \

# 		# train on scenes 2nd obj
# 	echo "python 04_decoding_single_ch.py -w \
# -s $sub --train-cond 'two_objects' --label All2ndObj \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --train-query \"Shape2+Colour2\" \
# --test-cond 'two_objects' \
# --test-query \"Shape1+Colour1\" "
# # --test-cond 'one_object' \
# # --test-query \"Shape1+Colour1\" \


# ## TRAIN OBJECTS ON SCENES (ORDER OF THE VISUAL SCENE)

# 		# RIGHT OBJECT
# 	echo "python 04_decoding_single_ch.py -w \
# --label AllRightObject -s $sub \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --train-cond 'two_objects' \
# --train-query \"Right_obj\" "
# # --test-cond 'one_object' \
# # --test-query \"Shape1+Colour1\" "
	
# 		# LEFT OBJECT
# 	echo "python 04_decoding_single_ch.py -w \
# --label AllLeftObject -s $sub \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --train-cond 'two_objects' \
# --train-query \"Left_obj\" "
# # --test-cond 'one_object' \
# # --test-query \"Shape1+Colour1\" "


# ## RELATION
# 	echo "python 04_decoding_single_ch.py -w \
# -s $sub --label Relation \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --train-cond 'two_objects' \
# --train-query \"Relation\" "

# ## FLASH
# 	echo "python 04_decoding_single_ch.py -w \
# -s $sub --label Flash \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --train-cond 'two_objects' \
# --train-query \"Flash\" "

# ## MATCHING
# 	echo "python 04_decoding_single_ch.py -w \
# -s $sub --label Matching \
# --train-cond 'two_objects' \
# --train-query \"Matching\" "

# ## BUTTON	
# 	echo "python 04_decoding_single_ch.py -w \
# -s $sub --label Button \
# --train-cond 'two_objects' \
# --train-query \"Button\" "

# ## PERF
# 	echo "python 04_decoding_single_ch.py -w \
# -s $sub --label Perf \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --train-cond 'two_objects' \
# --train-query \"Perf\" "




# ## USE WINDOWS AROUND EACH COND
# ## maybe later ? 

# 	## COLOUR
# 	# train on one object
# 	echo "python 04_decoding_single_ch.py -w \
# -s $sub --label winColour \
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
# 	echo "python 04_decoding_single_ch.py -w \
# -s $sub --label winColour1 \
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

# 	echo "python 04_decoding_single_ch.py -w \
# -s $sub --label winColour2 \
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
# 	echo "python 04_decoding_single_ch.py -w \
# -s $sub --label winShape \
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
# 	echo "python 04_decoding_single_ch.py -w \
# -s $sub --label winShape1 \
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

# 	echo "python 04_decoding_single_ch.py -w \
# -s $sub --label winShape2 \
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

# 	## relation
# 	echo "python 04_decoding_single_ch.py -w \
# -s $sub --label winRelation \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --train-cond 'two_objects' \
# --train-query \"Relation\" \
# --windows \"1.2,2.\" "

# 	## OBJECTS
# 	# train on all other trials, gen to first, then to 2nd object
# 	echo "python 04_decoding_single_ch.py -w \
# -s $sub --label winAllObject \
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
# 	echo "python 04_decoding_single_ch.py -w \
# -s $sub --label winAll1stObj \
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
# 	echo "python 04_decoding_single_ch.py -w \
# -s $sub --label winAll2ndObj \
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
# 	echo "python 04_decoding_single_ch.py -w \
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
# 	echo "python 04_decoding_single_ch.py -w \
# -s $sub --label winShapesdelay \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --train-cond 'two_objects' \
# --train-query \"Shape1\" \
# --windows \"3.5,5.\" \
# --train-cond 'two_objects' \
# --train-query \"Shape2\" \
# --windows \"3.5,5.\" "

# ## RELATION
# 	echo "python 04_decoding_single_ch.py -w \
# -s $sub --label winRelationdelay \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --train-cond 'two_objects' \
# --train-query \"Relation\" \
# --windows \"3.5,5.\" "

# ## COLORS
# 	echo "python 04_decoding_single_ch.py -w \
# -s $sub --label winColoursdelay \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --train-cond 'two_objects' \
# --train-query \"Colour1\" \
# --windows \"3.5,5.\" \
# --train-cond 'two_objects' \
# --train-query \"Colour2\" \
# --windows \"3.5,5.\" "


# 	echo "python 04_decoding_single_ch.py -w \
# -s $sub --label winSideObjectsdelay \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --train-cond 'two_objects' \
# --train-query \"Left_obj\" \
# --windows \"3.5,5.\" \
# --train-cond 'two_objects' \
# --train-query \"Right_obj\" \
# --windows \"3.5,5.\" "

# ## FLASH
# 	echo "python 04_decoding_single_ch.py -w \
# -s $sub --label winFlash \
# --windows \"4.5,6.\" \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --train-cond 'two_objects' \
# --train-query \"Flash\" "

# ## MATCHING
# 	echo "python 04_decoding_single_ch.py -w \
# -s $sub --label winMatching \
# --windows \"5.5,7.5\" \
# --train-cond 'two_objects' \
# --train-query \"Matching\" "

# ## BUTTON	
# 	echo "python 04_decoding_single_ch.py -w \
# -s $sub --label winButton \
# --windows \"5.5,7.5\" \
# --train-cond 'two_objects' \
# --train-query \"Button\" "

# ## PERF
# 	echo "python 04_decoding_single_ch.py -w \
# -s $sub --label winPerf \
# --windows \"6.,8.\" \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --train-cond 'two_objects' \
# --train-query \"Perf\" "



done