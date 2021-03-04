for sub in $(python configs/config.py)
do
	# for freq_band in low_high low_low high_low high_vlow low_vlow low_vhigh high_vhigh vhigh_vhigh vhigh_high vhigh_low
	# do

declare -a all_colours
all_colours[0]="rouge;bleu;vert"
all_colours[1]="vert;rouge;bleu"
all_colours[2]="bleu;vert;rouge"

## COLOURS
for colours in "${all_colours[@]}" # loop over all combinations of colours (the first one is the train label, the other two are for negatvie examples)
do
	IFS=";" read -r -a colours <<< "${colours}" # from string to array

	# train on localizer words only
	echo "python 04_decoding.py --train-cond 'localizer' --label Colour \
--test-cond 'one_object' --test-cond 'two_objects' -w --timegen -s $sub \
--train-query-1 \"Loc_word=='${colours[0]}'\" \
--train-query-2 \"Loc_word in ['${colours[1]}', '${colours[2]}']\" \
--test-query-1 \"Colour1=='${colours[0]}'\" \
--test-query-2 \"Colour1 in ['${colours[1]}', '${colours[2]}']\""

	# train on localizer words + images
	echo "python 04_decoding.py --train-cond 'localizer' --label Colour \
--test-cond 'one_object' --test-cond 'two_objects' -w --timegen -s $sub \
--train-query-1 \"Loc_word in ['${colours[0]}', 'img_${colours[0]}']\" \
--train-query-2 \"Loc_word in ['${colours[1]}', '${colours[2]}', 'img_${colours[1]}', 'img_${colours[2]}']\" \
--test-query-1 \"Colour1=='${colours[0]}'\" \
--test-query-2 \"Colour1 in ['${colours[1]}', '${colours[2]}']\""

	# train on one object
	echo "python 04_decoding.py \
 --train-cond 'one_object' --label Colour \
-w --timegen -s $sub --test-cond 'two_objects' \
--train-query-1 \"Colour1=='${colours[0]}'\" \
--train-query-2 \"Colour1 in ['${colours[1]}', '${colours[2]}']\" \
--test-query-1 \"Colour1=='${colours[0]}'\" \
--test-query-2 \"Colour1 in ['${colours[1]}', '${colours[2]}']\""

	# train on two objects
	echo "python 04_decoding.py \
 --train-cond 'two_objects' --label Colour \
-w --timegen -s $sub --test-cond 'one_object' \
--train-query-1 \"Colour1=='${colours[0]}'\" \
--train-query-2 \"Colour1 in ['${colours[1]}', '${colours[2]}']\" \
--test-query-1 \"Colour1=='${colours[0]}'\" \
--test-query-2 \"Colour1 in ['${colours[1]}', '${colours[2]}']\""	


done


#### SHAPES ####
declare -a all_shapes
all_shapes[0]="carre;cercle;triangle"
all_shapes[1]="triangle;carre;cercle"
all_shapes[2]="cercle;triangle;carre"

for shapes in "${all_shapes[@]}" # loop over all combinations of shapes (the first one is the train label, the other two are for negatvie examples)
do
	IFS=";" read -r -a shapes <<< "${shapes}" # from string to array

	# train on localizer words only
	echo "python 04_decoding.py --train-cond 'localizer' --label Shape \
--test-cond 'one_object' --test-cond 'two_objects' -w --timegen -s $sub \
--train-query-1 \"Loc_word=='${shapes[0]}'\" \
--train-query-2 \"Loc_word in ['${shapes[1]}', '${shapes[2]}']\" \
--test-query-1 \"Shape1=='${shapes[0]}'\" \
--test-query-2 \"Shape1 in ['${shapes[1]}', '${shapes[2]}']\""

	# train on localizer words + images
	echo "python 04_decoding.py --train-cond 'localizer' --label Shape \
--test-cond 'one_object' --test-cond 'two_objects' -w --timegen -s $sub \
--train-query-1 \"Loc_word in ['${shapes[0]}', 'img_${shapes[0]}']\" \
--train-query-2 \"Loc_word in ['${shapes[1]}', '${shapes[2]}', 'img_${shapes[1]}', 'img_${shapes[2]}']\" \
--test-query-1 \"Shape1=='${shapes[0]}'\" \
--test-query-2 \"Shape1 in ['${shapes[1]}', '${shapes[2]}']\""

	# train on one object
	echo "python 04_decoding.py \
 --train-cond 'one_object' --label Shape \
-w --timegen -s $sub --test-cond 'two_objects' \
--train-query-1 \"Shape1=='${shapes[0]}'\" \
--train-query-2 \"Shape1 in ['${shapes[1]}', '${shapes[2]}']\" \
--test-query-1 \"Shape1=='${shapes[0]}'\" \
--test-query-2 \"Shape1 in ['${shapes[1]}', '${shapes[2]}']\""

	# train on two objects
	echo "python 04_decoding.py \
 --train-cond 'two_objects' --label Shape \
-w --timegen -s $sub --test-cond 'one_object' \
--train-query-1 \"Shape1=='${shapes[0]}'\" \
--train-query-2 \"Shape1 in ['${shapes[1]}', '${shapes[2]}']\" \
--test-query-1 \"Shape1=='${shapes[0]}'\" \
--test-query-2 \"Shape1 in ['${shapes[1]}', '${shapes[2]}']\""

done



## TRAIN OBJECTS
for colour in rouge bleu vert
do
	for shape in carre cercle triangle
	do
			# train on all other trials, gen to first object
			echo "python 04_decoding.py \
--train-cond 'one_object' --label AllObject \
-w --timegen -s $sub --test-cond 'two_objects' \
--train-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
--train-query-2 \"Shape1!='$shape' or Colour1!='$colour'\" \
--test-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
--test-query-2 \"Shape1!='$shape' or Colour1!='$colour'\""

			# train on all other trials, gen to second object
			echo "python 04_decoding.py \
--train-cond 'one_object' --label All2ndObject \
-w --timegen -s $sub --test-cond 'two_objects' \
--train-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
--train-query-2 \"Shape1!='$shape' or Colour1!='$colour'\" \
--test-query-1 \"Shape2=='$shape' and Colour2=='$colour'\" \
--test-query-2 \"Shape2!='$shape' or Colour2!='$colour'\""



	## BINDING CONDITION

	# train on one color vs all others, always with the same shape
	# test 1st on the same with 2 objects (maybe should constrain the second object not to be the same shape/color?)
	# = triangle bleue vs other triangles.
	# 2nd test only on the cases where the second object is the same color
	# = triangle bleue vs other triangles, when the second object is bleue
	# 3rd is the same but when the second object is blue but not a triangle
			echo "python 04_decoding.py \
--train-cond 'one_object' --label ColorBinding \
-w --timegen -s $sub --test-cond 'two_objects' \
--train-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
--train-query-2 \"Shape1=='$shape' and Colour1!='$colour'\" \
--test-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
--test-query-2 \"Shape1=='$shape' and Colour1!='$colour'\" \
--test-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
--test-query-2 \"Shape1=='$shape' and Colour1!='$colour' and Colour2=='$colour'\" \
--test-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
--test-query-2 \"Shape1=='$shape' and Colour1!='$colour' and Colour2=='$colour' and Shape2!='$shape'\""


	# train on one shape vs all others, always with the same color
	# test 1st on the same with 2 objects
	# = triangle bleue vs other bleue stuff.
	# 2nd test only on the cases where the second object is the same shape
	# = triangle bleue vs other blue stuff, when the second object is a triangle
	# 3rd is the same but when the second object is a triangle but not blue 
			echo "python 04_decoding.py --train-cond 'one_object' --label ShapeBinding \
-w --timegen -s $sub --test-cond 'two_objects' \
--train-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
--train-query-2 \"Shape1!='$shape' and Colour1=='$colour'\" \
--test-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
--test-query-2 \"Shape1!='$shape' and Colour1=='$colour'\" \
--test-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
--test-query-2 \"Shape1!='$shape' and Colour1=='$colour' and Shape2=='$shape'\" \
--test-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
--test-query-2 \"Shape1!='$shape' and Colour1=='$colour' and Shape2=='$shape' and Colour2!='$colour'\" "




			## #MATCHING ONLY FROM TRANING 
			# train on all other trials, gen to first object
			echo "python 04_decoding.py \
--train-cond 'one_object' --label AllObject \
-w --timegen -s $sub --test-cond 'two_objects' \
--train-query-1 \"Shape1=='$shape' and Colour1=='$colour' and Matching=='match'\" \
--train-query-2 \"Shape1!='$shape' or Colour1!='$colour' and Matching=='match'\" \
--test-query-1 \"Shape1=='$shape' and Colour1=='$colour' and Matching=='match'\" \
--test-query-2 \"Shape1!='$shape' or Colour1!='$colour' and Matching=='match'\""

		# train on all other trials, gen to second object
			echo "python 04_decoding.py \
--train-cond 'one_object' --label All2ndObject \
-w --timegen -s $sub --test-cond 'two_objects' \
--train-query-1 \"Shape1=='$shape' and Colour1=='$colour' and Matching=='match'\" \
--train-query-2 \"Shape1!='$shape' or Colour1!='$colour' and Matching=='match'\" \
--test-query-1 \"Shape2=='$shape' and Colour2=='$colour' and Matching=='match'\" \
--test-query-2 \"Shape2!='$shape' or Colour2!='$colour' and Matching=='match'\""


			## # NON MATCHING ONLY
			# train on all other trials, gen to first object
			echo "python 04_decoding.py \
--train-cond 'one_object' --label AllObject \
-w --timegen -s $sub --test-cond 'two_objects' \
--train-query-1 \"Shape1=='$shape' and Colour1=='$colour' and Matching=='nonmatch'\" \
--train-query-2 \"Shape1!='$shape' or Colour1!='$colour' and Matching=='nonmatch'\" \
--test-query-1 \"Shape1=='$shape' and Colour1=='$colour' and Matching=='nonmatch'\" \
--test-query-2 \"Shape1!='$shape' or Colour1!='$colour' and Matching=='nonmatch'\""

		# train on all other trials, gen to second object
			echo "python 04_decoding.py \
--train-cond 'one_object' --label All2ndObject \
-w --timegen -s $sub --test-cond 'two_objects' \
--train-query-1 \"Shape1=='$shape' and Colour1=='$colour' and Matching=='nonmatch'\" \
--train-query-2 \"Shape1!='$shape' or Colour1!='$colour' and Matching=='nonmatch'\" \
--test-query-1 \"Shape2=='$shape' and Colour2=='$colour' and Matching=='nonmatch'\" \
--test-query-2 \"Shape2!='$shape' or Colour2!='$colour' and Matching=='nonmatch'\""

	done
done



## that is shit, we have some negative class that are positive on the test set ... 
# # IMG LOCALIZER  TRAIN ON IMAGES AND TEST ON WORDS
# for colour in rouge bleu vert
# do
# 	echo "python 04_decoding.py --train-cond 'localizer' --label Img2WordC \
# --test-cond 'localizer' -w --timegen -s $sub \
# --train-query-1 \"Loc_word=='img_$colour'\" \
# --train-query-2 \"Loc_word!='img_$colour'\" \
# --test-query-1 \"Loc_word=='$colour'\" \
# --test-query-2 \"Loc_word!='$colour'\""
# done

# for shape in carre cercle triangle
# do
# 	echo "python 04_decoding.py --train-cond 'localizer' --label Img2WordS \
# --test-cond 'localizer' -w --timegen -s $sub \
# --train-query-1 \"Loc_word=='img_$shape'\" \
# --train-query-2 \"Loc_word!='img_$shape'\" \
# --test-query-1 \"Loc_word=='$shape'\" \
# --test-query-2 \"Loc_word!='$shape'\""
# done


# # IMG LOCALIZER  TRAIN ON WORDS AND TEST ON IMAGES
# for colour in rouge bleu vert
# do
# 	echo "python 04_decoding.py --train-cond 'localizer' --label Word2ImgC \
# --test-cond 'localizer' -w --timegen -s $sub \
# --train-query-1 \"Loc_word=='$colour'\" \
# --train-query-2 \"Loc_word!='$colour'\" \
# --test-query-1 \"Loc_word=='img_$colour'\" \
# --test-query-2 \"Loc_word!='img_$colour'\""
# done

# for shape in carre cercle triangle
# do
# 	echo "python 04_decoding.py --train-cond 'localizer' --label Word2ImgS \
# --test-cond 'localizer' -w --timegen -s $sub \
# --train-query-1 \"Loc_word=='$shape'\" \
# --train-query-2 \"Loc_word!='$shape'\" \
# --test-query-1 \"Loc_word=='img_$shape'\" \
# --test-query-2 \"Loc_word!='img_$shape'\""
# done


# # --localizer --path2loc 'Single_Chan_vs15/CMR_clean_sent' --pval-thresh 0.01 \
	# done
done