for sub in $(python configs/config.py)
do
	# for freq_band in low_high low_low high_low high_vlow low_vlow low_vhigh high_vhigh vhigh_vhigh vhigh_high vhigh_low
	# do

## MISMTACHES
# 	## TRAIN ON ONE OBJECT
# 	echo "python 04_decoding.py \
# --train-cond 'one_object' --label KindMismatch \
# --timegen -s $sub \
# --train-query-1 \"Matching=='match'\" \
# --train-query-2 \"Matching=='nonmatch'\" \
# --split-queries \"Matching=='match' or Error_type=='colour'\" \
# --split-queries \"Matching=='match' or Error_type=='shape'\" \
# --test-cond 'two_objects' \
# --test-query-1 \"Matching=='match'\" \
# --test-query-2 \"Matching=='nonmatch'\" "


	echo "python 04_decoding.py \
--train-cond 'one_object' --label ColourMismatch \
--timegen -s $sub \
--train-query-1 \"Matching=='match'\" \
--train-query-2 \"Error_type=='colour'\" "

echo "python 04_decoding.py \
--train-cond 'one_object' --label ShapeMismatch \
--timegen -s $sub \
--train-query-1 \"Matching=='match'\" \
--train-query-2 \"Error_type=='shape'\" "

	## TRAIN ON TWO OBJECTS
	echo "python 04_decoding.py \
--train-cond 'two_objects' --label KindMismatch \
--timegen -s $sub \
--train-query-1 \"Matching=='match'\" \
--train-query-2 \"Matching=='nonmatch'\" \
--split-queries \"Matching=='match' or Error_type=='l0'\" \
--split-queries \"Matching=='match' or Error_type=='l1'\" \
--split-queries \"Matching=='match' or Error_type=='l2'\" "

	echo "python 04_decoding.py \
--train-cond 'two_objects' --label PropMismatch \
--timegen -s $sub \
--train-query-1 \"Matching=='match'\" \
--train-query-2 \"Error_type=='l0'\" "

	echo "python 04_decoding.py \
--train-cond 'two_objects' --label BindMismatch \
--timegen -s $sub \
--train-query-1 \"Matching=='match'\" \
--train-query-2 \"Error_type=='l1'\" "

	echo "python 04_decoding.py \
--train-cond 'two_objects' --label RelMismatch \
--timegen -s $sub \
--train-query-1 \"Matching=='match'\" \
--train-query-2 \"Error_type=='l2'\" "



## DECODE BUTTON PRESSED
	echo "python 04_decoding.py \
--train-cond 'one_object' --label Button \
--timegen -s $sub \
--train-query-1 \"Button=='left'\" \
--train-query-2 \"Button=='right'\" \
--split-queries \"Matching=='match'\" \
--split-queries \"Matching=='nonmatch'\" \
--test-cond 'two_objects' \
--test-query-1 \"Button=='left'\" \
--test-query-2 \"Button=='right'\" "
# --split-queries \"Perf==1\" \

	echo "python 04_decoding.py \
--train-cond 'two_objects' --label Button \
--timegen -s $sub \
--train-query-1 \"Button=='left'\" \
--train-query-2 \"Button=='right'\" \
--split-queries \"Matching=='match'\" \
--split-queries \"Matching=='nonmatch'\" \
--test-cond 'one_object' \
--test-query-1 \"Button=='left'\" \
--test-query-2 \"Button=='right'\" "
# --split-queries \"Perf==1\" \


## FLASH 
	echo "python 04_decoding.py \
--train-cond 'two_objects' --label Flash \
--timegen -s $sub \
--train-query-1 \"Flash==True\" \
--train-query-2 \"Flash==False\" "
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --split-queries \"Perf==1\" "




# declare -a all_colours
# all_colours[0]="rouge;bleu;vert"
# all_colours[1]="vert;rouge;bleu"
# all_colours[2]="bleu;vert;rouge"

# ## COLOURS
# for colours in "${all_colours[@]}" # loop over all combinations of colours (the first one is the train label, the other two are for negatvie examples)
# do
# 	IFS=";" read -r -a colours <<< "${colours}" # from string to array

# # 	# train on localizer words only
# # 	echo "python 04_decoding.py --train-cond 'localizer' --label Colour \
# # --timegen -s $sub \
# # --train-query-1 \"Loc_word=='${colours[0]}'\" \
# # --train-query-2 \"Loc_word in ['${colours[1]}', '${colours[2]}']\" \
# # --test-cond 'one_object' \
# # --test-query-1 \"Colour1=='${colours[0]}'\" \
# # --test-query-2 \"Colour1 in ['${colours[1]}', '${colours[2]}']\" \
# # --test-cond 'two_objects' \
# # --test-query-1 \"Colour1=='${colours[0]}'\" \
# # --test-query-2 \"Colour1 in ['${colours[1]}', '${colours[2]}']\" "

# 	# train on localizer words + images
# 	echo "python 04_decoding.py --train-cond 'localizer' --label Colour \
# --timegen -s $sub \
# --train-query-1 \"Loc_word in ['${colours[0]}', 'img_${colours[0]}']\" \
# --train-query-2 \"Loc_word in ['${colours[1]}', '${colours[2]}', 'img_${colours[1]}', 'img_${colours[2]}']\" \
# --test-cond 'one_object' \
# --test-query-1 \"Colour1=='${colours[0]}'\" \
# --test-query-2 \"Colour1 in ['${colours[1]}', '${colours[2]}']\" \
# --test-cond 'two_objects' \
# --test-query-1 \"Colour1=='${colours[0]}'\" \
# --test-query-2 \"Colour1 in ['${colours[1]}', '${colours[2]}']\" "

# 	# train on one object
# 	echo "python 04_decoding.py \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --train-cond 'one_object' --label Colour \
# --timegen -s $sub \
# --train-query-1 \"Colour1=='${colours[0]}'\" \
# --train-query-2 \"Colour1 in ['${colours[1]}', '${colours[2]}']\" \
# --test-cond 'two_objects' \
# --test-query-1 \"Colour1=='${colours[0]}'\" \
# --test-query-2 \"Colour1 in ['${colours[1]}', '${colours[2]}']\""

# 	# train on two objects 1st
# 	echo "python 04_decoding.py \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --train-cond 'two_objects' --label Colour1 \
# --timegen -s $sub \
# --train-query-1 \"Colour1=='${colours[0]}'\" \
# --train-query-2 \"Colour1 in ['${colours[1]}', '${colours[2]}']\" \
# --test-cond 'one_object' \
# --test-query-1 \"Colour1=='${colours[0]}'\" \
# --test-query-2 \"Colour1 in ['${colours[1]}', '${colours[2]}']\""	

# 	# train on two objects 2nd
# 	echo "python 04_decoding.py \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --train-cond 'two_objects' --label Colour2 \
# --timegen -s $sub \
# --train-query-1 \"Colour2=='${colours[0]}'\" \
# --train-query-2 \"Colour2 in ['${colours[1]}', '${colours[2]}']\" \
# --test-cond 'one_object' \
# --test-query-1 \"Colour2=='${colours[0]}'\" \
# --test-query-2 \"Colour2 in ['${colours[1]}', '${colours[2]}']\""	


# done


# #### SHAPES ####
# declare -a all_shapes
# all_shapes[0]="carre;cercle;triangle"
# all_shapes[1]="triangle;carre;cercle"
# all_shapes[2]="cercle;triangle;carre"

# for shapes in "${all_shapes[@]}" # loop over all combinations of shapes (the first one is the train label, the other two are for negatvie examples)
# do
# 	IFS=";" read -r -a shapes <<< "${shapes}" # from string to array

# # 	# train on localizer words only
# # 	echo "python 04_decoding.py --train-cond 'localizer' --label Shape \
# # --timegen -s $sub \
# # --train-query-1 \"Loc_word=='${shapes[0]}'\" \
# # --train-query-2 \"Loc_word in ['${shapes[1]}', '${shapes[2]}']\" \
# # --test-cond 'one_object' \
# # --test-query-1 \"Shape1=='${shapes[0]}'\" \
# # --test-query-2 \"Shape1 in ['${shapes[1]}', '${shapes[2]}']\" \
# # --test-cond 'two_objects' \
# # --test-query-1 \"Shape1=='${shapes[0]}'\" \
# # --test-query-2 \"Shape1 in ['${shapes[1]}', '${shapes[2]}']\" "

# 	# train on localizer words + images
# 	echo "python 04_decoding.py --train-cond 'localizer' --label Shape \
# --timegen -s $sub \
# --train-query-1 \"Loc_word in ['${shapes[0]}', 'img_${shapes[0]}']\" \
# --train-query-2 \"Loc_word in ['${shapes[1]}', '${shapes[2]}', 'img_${shapes[1]}', 'img_${shapes[2]}']\" \
# --test-cond 'one_object' \
# --test-query-1 \"Shape1=='${shapes[0]}'\" \
# --test-query-2 \"Shape1 in ['${shapes[1]}', '${shapes[2]}']\" \
# --test-cond 'two_objects' \
# --test-query-1 \"Shape1=='${shapes[0]}'\" \
# --test-query-2 \"Shape1 in ['${shapes[1]}', '${shapes[2]}']\" "

# 	# train on one object
# 	echo "python 04_decoding.py \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --train-cond 'one_object' --label Shape \
# --timegen -s $sub --test-cond 'two_objects' \
# --train-query-1 \"Shape1=='${shapes[0]}'\" \
# --train-query-2 \"Shape1 in ['${shapes[1]}', '${shapes[2]}']\" \
# --test-query-1 \"Shape1=='${shapes[0]}'\" \
# --test-query-2 \"Shape1 in ['${shapes[1]}', '${shapes[2]}']\""

# 	# train on two objects 1st
# 	echo "python 04_decoding.py \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --train-cond 'two_objects' --label Shape1 \
# --timegen -s $sub --test-cond 'one_object' \
# --train-query-1 \"Shape1=='${shapes[0]}'\" \
# --train-query-2 \"Shape1 in ['${shapes[1]}', '${shapes[2]}']\" \
# --test-query-1 \"Shape1=='${shapes[0]}'\" \
# --test-query-2 \"Shape1 in ['${shapes[1]}', '${shapes[2]}']\""

# 	# train on two objects
# 	echo "python 04_decoding.py \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --train-cond 'two_objects' --label Shape2 \
# --timegen -s $sub --test-cond 'one_object' \
# --train-query-1 \"Shape2=='${shapes[0]}'\" \
# --train-query-2 \"Shape2 in ['${shapes[1]}', '${shapes[2]}']\" \
# --test-query-1 \"Shape2=='${shapes[0]}'\" \
# --test-query-2 \"Shape2 in ['${shapes[1]}', '${shapes[2]}']\""

# done


# # for colours in "${all_colours[@]}" # loop over all combinations of colours (the first one is the train label, the other two are for negatvie examples)
# # do
# # 	IFS=";" read -r -a colours <<< "${colours}" # from string to array
	
# # 	for shapes in "${all_shapes[@]}" # loop over all combinations of shapes (the first one is the train label, the other two are for negatvie examples)
# # 	do
# # 		IFS=";" read -r -a shapes <<< "${shapes}" # from string to array

# # 	## COLOUR ON OBJECTS WITH LEAVE ON SHAPE OUT
# # 	echo "python 04_decoding.py --timegen -s $sub --label ColourShapeLOO \
# # --split-queries \"Matching=='match'\" \
# # --split-queries \"Matching=='nonmatch'\" \
# # --train-cond 'one_object' \
# # --train-query-1 \"Shape1 in ['${shapes[0]}', '${shapes[1]}'] and Colour1=='${colours[0]}'\" \
# # --train-query-2 \"Shape1 in ['${shapes[0]}', '${shapes[1]}'] and Colour1!='${colours[0]}'\" \
# # --test-cond 'one_object' \
# # --test-query-1 \"Shape1=='${shapes[2]}' and Colour1=='${colours[0]}'\"  \
# # --test-query-2 \"Shape1=='${shapes[2]}' and Colour1!='${colours[0]}'\"  \
# # --test-cond 'two_objects' \
# # --test-query-1 \"Shape1=='${shapes[2]}' and Colour1=='${colours[0]}'\"  \
# # --test-query-2 \"Shape1=='${shapes[2]}' and Colour1!='${colours[0]}'\"  \
# # --test-cond 'two_objects' \
# # --test-query-1 \"Shape2=='${shapes[2]}' and Colour2=='${colours[0]}'\"  \
# # --test-query-2 \"Shape2=='${shapes[2]}' and Colour2!='${colours[0]}'\"  "


# # 	## SHAPE ON OBJECTS WITH LEAVE ON COLOUR OUT
# # 	echo "python 04_decoding.py --timegen -s $sub --label ShapeColourLOO \
# # --split-queries \"Matching=='match'\" \
# # --split-queries \"Matching=='nonmatch'\" \
# # --train-cond 'one_object' \
# # --train-query-1 \"Colour1 in ['${colours[0]}', '${colours[1]}'] and Shape1=='${shapes[0]}'\" \
# # --train-query-2 \"Colour1 in ['${colours[0]}', '${colours[1]}'] and Shape1!='${shapes[0]}'\" \
# # --test-cond 'one_object' \
# # --test-query-1 \"Colour1=='${colours[2]}' and Shape1=='${shapes[0]}'\"  \
# # --test-query-2 \"Colour1=='${colours[2]}' and Shape1!='${shapes[0]}'\"  \
# # --test-cond 'two_objects' \
# # --test-query-1 \"Colour1=='${colours[2]}' and Shape1=='${shapes[0]}'\"  \
# # --test-query-2 \"Colour1=='${colours[2]}' and Shape1!='${shapes[0]}'\"  \
# # --test-cond 'two_objects' \
# # --test-query-1 \"Colour2=='${colours[2]}' and Shape2=='${shapes[0]}'\"  \
# # --test-query-2 \"Colour2=='${colours[2]}' and Shape2!='${shapes[0]}'\"  "

# # 	done
# # done



# ## TRAIN OBJECTS (ORDER OF THE SENTENCE)
# for colour in rouge bleu vert
# do
# 	for shape in carre cercle triangle
# 	do

# 			# train on all other trials, gen to first, then to 2nd object
# 			echo "python 04_decoding.py \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --timegen -s $sub --train-cond 'one_object' --label AllObject \
# --train-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
# --train-query-2 \"Shape1!='$shape' or Colour1!='$colour'\" \
# --test-cond 'two_objects' \
# --test-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
# --test-query-2 \"Shape1!='$shape' or Colour1!='$colour'\" \
# --test-cond 'two_objects' \
# --test-query-1 \"Shape2=='$shape' and Colour2=='$colour'\" \
# --test-query-2 \"Shape2!='$shape' or Colour2!='$colour'\" \
# --test-cond 'two_objects' \
# --test-query-1 \"Right_obj=='${shape}_$colour'\" \
# --test-query-2 \"Right_obj!='${shape}_$colour'\" \
# --test-cond 'two_objects' \
# --test-query-1 \"Left_obj=='${shape}_$colour'\" \
# --test-query-2 \"Left_obj!='${shape}_$colour'\" "


# 			# train on first, gen to 2nd object
# 			echo "python 04_decoding.py \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --timegen -s $sub --train-cond 'two_objects' --label All1stObj \
# --train-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
# --train-query-2 \"Shape1!='$shape' or Colour1!='$colour'\" \
# --test-cond 'one_object' \
# --test-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
# --test-query-2 \"Shape1!='$shape' or Colour1!='$colour'\" "
# # --test-cond 'two_objects' \ 
# # --test-query-1 \"Shape2=='$shape' and Colour2=='$colour'\" \
# # --test-query-2 \"Shape2!='$shape' or Colour2!='$colour'\" " # probably negative class issue

# 			# train on all 2nd obj trial trials, gen to first
# 			echo "python 04_decoding.py \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --timegen -s $sub --train-cond 'two_objects' --label All2ndObj \
# --train-query-1 \"Shape2=='$shape' and Colour2=='$colour'\" \
# --train-query-2 \"Shape2!='$shape' or Colour2!='$colour'\" \
# --test-cond 'one_object' \
# --test-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
# --test-query-2 \"Shape1!='$shape' or Colour1!='$colour'\" "
# # --test-cond 'two_objects' \
# # --test-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
# # --test-query-2 \"Shape1!='$shape' or Colour1!='$colour'\" " # probably negative class issue


# 			# train on completely first, gen to 2nd object
# 			echo "python 04_decoding.py \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --timegen -s $sub --train-cond 'two_objects' --label AllComplDiffObjScenes \
# --train-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
# --train-query-2 \"Shape1!='$shape' and Colour1!='$colour'\" \
# --test-cond 'one_object' \
# --test-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
# --test-query-2 \"Shape1!='$shape' and Colour1!='$colour'\" "
# # --test-cond 'two_objects' \
# # --test-query-1 \"Shape2=='$shape' and Colour2=='$colour'\" \
# # --test-query-2 \"Shape2!='$shape' and Colour2!='$colour'\" " # probably negative class issue

# # 			# train on all completely 2nd obj trial trials, gen to first
# # 			echo "python 04_decoding.py \
# # --split-queries \"Matching=='match'\" \
# # --split-queries \"Matching=='nonmatch'\" \
# # --timegen -s $sub --train-cond 'two_objects' --label AllComplDiff2ndObjScenes \
# # --train-query-1 \"Shape2=='$shape' and Colour2=='$colour'\" \
# # --train-query-2 \"Shape2!='$shape' and Colour2!='$colour'\" \
# # --test-cond 'one_object' \
# # --test-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
# # --test-query-2 \"Shape1!='$shape' and Colour1!='$colour'\" \
# # --test-cond 'two_objects' \
# # --test-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
# # --test-query-2 \"Shape1!='$shape' and Colour1!='$colour'\" "
# # # --test-cond 'two_objects' \
# # # --test-query-1 \"Right_obj=='${shape}_$colour'\" \
# # # --test-query-2 \"Right_obj!='${shape}_$colour'\" \
# # # --test-cond 'two_objects' \
# # # --test-query-1 \"Left_obj=='${shape}_$colour'\" \
# # # --test-query-2 \"Left_obj!='${shape}_$colour'\" "

# 			## Individual objects, train on object that are COMPLETELY different (!= in both features)
# 			echo "python 04_decoding.py \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --timegen -s $sub --train-cond 'one_object' --label AllComplDiffObject \
# --train-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
# --train-query-2 \"Shape1!='$shape' and Colour1!='$colour'\" \
# --test-cond 'two_objects' \
# --test-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
# --test-query-2 \"Shape1!='$shape' and Colour1!='$colour'\" \
# --test-cond 'two_objects' \
# --test-query-1 \"Shape2=='$shape' and Colour2=='$colour'\" \
# --test-query-2 \"Shape2!='$shape' and Colour2!='$colour'\""


# 			## Individual objects, train on object that are similar in one feature
# 			# Should you test on all ? to compare to the one above?
# 			echo "python 04_decoding.py \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --timegen -s $sub --train-cond 'one_object' --label AllSmwtDiffObject \
# --train-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
# --train-query-2 \"(Shape1=='$shape' or Colour1=='$colour') and not (Shape1=='$shape' and Colour1=='$colour') \" \
# --test-cond 'two_objects' \
# --test-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
# --test-query-2 \"(Shape1=='$shape' or Colour1=='$colour') and not (Shape1=='$shape' and Colour1=='$colour')\" \
# --test-cond 'two_objects' \
# --test-query-1 \"Shape2=='$shape' and Colour2=='$colour'\" \
# --test-query-2 \"(Shape2=='$shape' or Colour2=='$colour') and not (Shape2=='$shape' and Colour2=='$colour')\""


# # 	## BINDING CONDITION

# # 	# train on one color vs all others, always with the same shape
# # 	# test 1st on the same with 2 objects (maybe should constrain the second object not to be the same shape/color?)
# # 	# = triangle bleue vs other triangles.
# # 	# 2nd test only on the cases where the second object is the same color
# # 	# = triangle bleue vs other triangles, when the second object is bleue
# # 	# 3rd is the same but when the second object is blue but not a triangle
# # 			echo "python 04_decoding.py \
# # --split-queries \"Matching=='match'\" \
# # --split-queries \"Matching=='nonmatch'\" \
# # --timegen -s $sub --train-cond 'one_object' --label ColorBinding \
# # --train-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
# # --train-query-2 \"Shape1=='$shape' and Colour1!='$colour'\" \
# # --test-cond 'two_objects' \
# # --test-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
# # --test-query-2 \"Shape1=='$shape' and Colour1!='$colour'\" \
# # --test-cond 'two_objects' \
# # --test-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
# # --test-query-2 \"Shape1=='$shape' and Colour1!='$colour' and Colour2=='$colour'\" \
# # --test-cond 'two_objects' \
# # --test-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
# # --test-query-2 \"Shape1=='$shape' and Colour1!='$colour' and Colour2=='$colour' and Shape2!='$shape'\""


# # 	# train on one shape vs all others, always with the same color
# # 	# test 1st on the same with 2 objects
# # 	# = triangle bleue vs other bleue stuff.
# # 	# 2nd test only on the cases where the second object is the same shape
# # 	# = triangle bleue vs other blue stuff, when the second object is a triangle
# # 	# 3rd is the same but when the second object is a triangle but not blue 
# # 			echo "python 04_decoding.py \
# # --split-queries \"Matching=='match'\" \
# # --split-queries \"Matching=='nonmatch'\" \
# # --timegen -s $sub --train-cond 'one_object' --label ShapeBinding \
# # --train-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
# # --train-query-2 \"Shape1!='$shape' and Colour1=='$colour'\" \
# # --test-cond 'two_objects' \
# # --test-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
# # --test-query-2 \"Shape1!='$shape' and Colour1=='$colour'\" \
# # --test-cond 'two_objects' \
# # --test-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
# # --test-query-2 \"Shape1!='$shape' and Colour1=='$colour' and Shape2=='$shape'\" \
# # --test-cond 'two_objects' \
# # --test-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
# # --test-query-2 \"Shape1!='$shape' and Colour1=='$colour' and Shape2=='$shape' and Colour2!='$colour'\" "




# 	done
# done




# ## TRAIN OBJECTS ON SCENES (ORDER OF THE VISUAL SCENE)
# for colour in rouge bleu vert
# do
# 	for shape in carre cercle triangle
# 	do
# 		# RIGHT OBJECT
# 		echo "python 04_decoding.py \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --label AllRightObject --timegen -s $sub \
# --train-cond 'two_objects' \
# --train-query-1 \"Right_obj=='${shape}_$colour'\" \
# --train-query-2 \"Right_obj!='${shape}_$colour'\" \
# --test-cond 'one_object' \
# --test-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
# --test-query-2 \"Shape1!='$shape' or Colour1!='$colour'\" "
	
# 		# LEFT OBJECT
# 		echo "python 04_decoding.py \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --label AllLeftObject --timegen -s $sub \
# --train-cond 'two_objects' \
# --train-query-1 \"Left_obj=='${shape}_$colour'\" \
# --train-query-2 \"Left_obj!='${shape}_$colour'\" \
# --test-cond 'one_object' \
# --test-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
# --test-query-2 \"Shape1!='$shape' or Colour1!='$colour'\" "


# # 	# SAME WITH THE NEGATIVE CLASS THAT CANT BE THE SAME AS THE POSITIVE CLASS
# # 		# RIGHT OBJECT
# # 		echo "python 04_decoding.py \
# # --split-queries \"Matching=='match'\" \
# # --split-queries \"Matching=='nonmatch'\" \
# # --label AllRightObjNotLeft --timegen -s $sub \
# # --train-cond 'two_objects' \
# # --train-query-1 \"Right_obj=='${shape}_$colour'\" \
# # --train-query-2 \"Right_obj!='${shape}_$colour' and Left_obj!='${shape}_$colour'\" \
# # --test-cond 'one_object' \
# # --test-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
# # --test-query-2 \"Shape1!='$shape' or Colour1!='$colour'\" "

# # 		# LEFT OBJECT
# # 		echo "python 04_decoding.py \
# # --split-queries \"Matching=='match'\" \
# # --split-queries \"Matching=='nonmatch'\" \
# # --label AllLeftObjNotRight --timegen -s $sub \
# # --train-cond 'two_objects' \
# # --train-query-1 \"Left_obj=='${shape}_$colour'\" \
# # --train-query-2 \"Left_obj!='${shape}_$colour' and Right_obj!='${shape}_$colour'\" \
# # --test-cond 'one_object' \
# # --test-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
# # --test-query-2 \"Shape1!='$shape' or Colour1!='$colour'\" "

# done
# done
 	


# SAME OBJECT
echo "python 04_decoding.py \
--split-queries \"Matching=='match'\" \
--split-queries \"Matching=='nonmatch'\" \
--split-queries \"Flash==1\" \
--split-queries \"Flash==0\" \
--label SameObject --timegen -s $sub \
--train-cond 'two_objects' \
--train-query-1 \"Left_obj==Right_obj\" \
--train-query-2 \"Left_obj!=Right_obj\" "

# SAME COLOUR
echo "python 04_decoding.py \
--timegen -s $sub --label SameColour \
--train-cond 'two_objects' \
--train-query-1 \"Colour1==Colour2\" \
--train-query-2 \"Colour1!=Colour2\" \
--split-queries \"Matching=='match'\" \
--split-queries \"Matching=='nonmatch'\" \
--split-queries \"Flash==1\" \
--split-queries \"Flash==0\" \
--test-cond 'two_objects' \
--test-query-1 \"Shape1==Shape2\" \
--test-query-2 \"Shape1!=Shape2\" "

# SAME SHAPE
echo "python 04_decoding.py \
--label SameShape --timegen -s $sub \
--train-cond 'two_objects' \
--train-query-1 \"Shape1==Shape2\" \
--train-query-2 \"Shape1!=Shape2\" \
--split-queries \"Matching=='match'\" \
--split-queries \"Matching=='nonmatch'\" \
--split-queries \"Flash==1\" \
--split-queries \"Flash==0\" \
--test-cond 'two_objects' \
--test-query-1 \"Colour1==Colour2\" \
--test-query-2 \"Colour1!=Colour2\" "





## RELATION
echo "python 04_decoding.py \
--split-queries \"Matching=='match'\" \
--split-queries \"Matching=='nonmatch'\" \
--split-queries \"Flash==1\" \
--split-queries \"Flash==0\" \
--label Relation --timegen -s $sub \
--train-cond 'two_objects' \
--train-query-1 \"Relation=='à gauche d\''\" \
--train-query-2 \"Relation=='à droite d\''\" "











## that is shit, we have some negative class that are positive on the test set ... 
# # IMG LOCALIZER  TRAIN ON IMAGES AND TEST ON WORDS
# for colour in rouge bleu vert
# do
# 	echo "python 04_decoding.py --train-cond 'localizer' --label Img2WordC \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --test-cond 'localizer' --timegen -s $sub \
# --train-query-1 \"Loc_word=='img_$colour'\" \
# --train-query-2 \"Loc_word!='img_$colour'\" \
# --test-query-1 \"Loc_word=='$colour'\" \
# --test-query-2 \"Loc_word!='$colour'\""
# done

# for shape in carre cercle triangle
# do
# 	echo "python 04_decoding.py --train-cond 'localizer' --label Img2WordS \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --test-cond 'localizer' --timegen -s $sub \
# --train-query-1 \"Loc_word=='img_$shape'\" \
# --train-query-2 \"Loc_word!='img_$shape'\" \
# --test-query-1 \"Loc_word=='$shape'\" \
# --test-query-2 \"Loc_word!='$shape'\""
# done


# # IMG LOCALIZER  TRAIN ON WORDS AND TEST ON IMAGES
# for colour in rouge bleu vert
# do
# 	echo "python 04_decoding.py --train-cond 'localizer' --label Word2ImgC \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --test-cond 'localizer' --timegen -s $sub \
# --train-query-1 \"Loc_word=='$colour'\" \
# --train-query-2 \"Loc_word!='$colour'\" \
# --test-query-1 \"Loc_word=='img_$colour'\" \
# --test-query-2 \"Loc_word!='img_$colour'\""
# done

# for shape in carre cercle triangle
# do
# 	echo "python 04_decoding.py --train-cond 'localizer' --label Word2ImgS \
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --test-cond 'localizer' --timegen -s $sub \
# --train-query-1 \"Loc_word=='$shape'\" \
# --train-query-2 \"Loc_word!='$shape'\" \
# --test-query-1 \"Loc_word=='img_$shape'\" \
# --test-query-2 \"Loc_word!='img_$shape'\""
# done


# # --localizer --path2loc 'Single_Chan_vs15/CMR_clean_sent' --pval-thresh 0.01 \
	# done
done