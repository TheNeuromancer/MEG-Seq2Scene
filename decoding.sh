for subject in theo
do
	# for freq_band in low_high low_low high_low high_vlow low_vlow low_vhigh high_vhigh vhigh_vhigh vhigh_high vhigh_low
	# do

for colour in rouge bleu vert
do
	# train on localizer
	echo "python decoding.py --baseline -v 6 --clip --smooth 5 \
--crossval 'kfold' --n_folds 5 --train-cond 'localizer' --label Colour \
--test-cond 'one_object' --test-cond 'two_objects' \
-w --timegen -s $subject --sfreq 100 \
--train-query-1 \"Loc_word=='$colour'\" \
--train-query-2 \"Loc_word!='$colour'\" \
--test-query-1 \"Colour1=='$colour'\" \
--test-query-2 \"Colour1!='$colour'\" \
-i \"Data/Epochs2/\" -o \"/Epochs/\" --tmin '-0.5' --tmax 7. "

	# train on imglocalizer
	echo "python decoding.py --baseline -v 6 --clip --smooth 5 \
--crossval 'kfold' --n_folds 5 --train-cond 'imgloc' --label Colour \
--test-cond 'one_object' --test-cond 'two_objects' \
-w --timegen -s $subject --sfreq 100 \
--train-query-1 \"Loc_word=='$colour'\" \
--train-query-2 \"Loc_word!='$colour'\" \
--test-query-1 \"Colour1=='$colour'\" \
--test-query-2 \"Colour1!='$colour'\" \
-i \"Data/Epochs2/\" -o \"/Epochs/\" --tmin '-0.5' --tmax 7. "

	# train on one object
	echo "python decoding.py --baseline -v 6 --clip --smooth 5 \
--crossval 'kfold' --n_folds 5 --train-cond 'one_object' --label Colour \
-w --timegen -s $subject --sfreq 100 --test-cond 'two_objects' \
--train-query-1 \"Colour1=='$colour'\" \
--train-query-2 \"Colour1!='$colour'\" \
--test-query-1 \"Colour1=='$colour'\" \
--test-query-2 \"Colour1!='$colour'\" \
-i \"Data/Epochs2/\" -o \"/Epochs/\" --tmin '-0.5' --tmax 7. "

done


for shape in carre cercle triangle
do
	# train on localizer
	echo "python decoding.py --baseline -v 6 --clip --smooth 5 \
--crossval 'kfold' --n_folds 5 --train-cond 'localizer' --label Shape \
--test-cond 'one_object' --test-cond 'two_objects' \
-w --timegen -s $subject --sfreq 100 \
--train-query-1 \"Loc_word=='$shape'\" \
--train-query-2 \"Loc_word!='$shape'\" \
--test-query-1 \"Shape1=='$shape'\" \
--test-query-2 \"Shape1!='$shape'\" \
-i \"Data/Epochs2/\" -o \"/Epochs/\" --tmin '-0.5' --tmax 7. "

	# train on imglocalizer
	echo "python decoding.py --baseline -v 6 --clip --smooth 5 \
--crossval 'kfold' --n_folds 5 --train-cond 'imgloc' --label Shape \
--test-cond 'one_object' --test-cond 'two_objects' \
-w --timegen -s $subject --sfreq 100 \
--train-query-1 \"Loc_word=='$shape'\" \
--train-query-2 \"Loc_word!='$shape'\" \
--test-query-1 \"Shape1=='$shape'\" \
--test-query-2 \"Shape1!='$shape'\" \
-i \"Data/Epochs2/\" -o \"/Epochs/\" --tmin '-0.5' --tmax 7. "

	# train on one object
	echo "python decoding.py --baseline -v 6 --clip --smooth 5 \
--crossval 'kfold' --n_folds 5 --train-cond 'one_object' --label Shape \
-w --timegen -s $subject --sfreq 100 --test-cond 'two_objects' \
--train-query-1 \"Shape1=='$shape'\" \
--train-query-2 \"Shape1!='$shape'\" \
--test-query-1 \"Shape1=='$shape'\" \
--test-query-2 \"Shape1!='$shape'\" \
-i \"Data/Epochs2/\" -o \"/Epochs/\" --tmin '-0.5' --tmax 7. "
done



## TRAIN OBJECTS
for colour in rouge bleu vert
do
	for shape in carre cercle triangle
	do

			# train on all other trials, gen to first object
			echo "python decoding.py --baseline -v 6 --clip --smooth 5 \
--crossval 'kfold' --n_folds 5 --train-cond 'one_object' --label AllObject \
-w --timegen -s $subject --sfreq 100 --test-cond 'two_objects' \
--train-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
--train-query-2 \"Shape1!='$shape' or Colour1!='$colour'\" \
--test-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
--test-query-2 \"\"Shape1!='$shape' or Colour1!='$colour'\" \
-i \"Data/Epochs2/\" -o \"/Epochs/\" --tmin '-0.5' --tmax 7. "

		# train on all other trials, gen to second object
			echo "python decoding.py --baseline -v 6 --clip --smooth 5 \
--crossval 'kfold' --n_folds 5 --train-cond 'one_object' --label All2ndObject \
-w --timegen -s $subject --sfreq 100 --test-cond 'two_objects' \
--train-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
--train-query-2 \"Shape1!='$shape' or Colour1!='$colour'\" \
--test-query-1 \"Shape2=='$shape' and Colour2=='$colour'\" \
--test-query-2 \"\"Shape2!='$shape' or Colour2!='$colour'\" \
-i \"Data/Epochs2/\" -o \"/Epochs/\" --tmin '-0.5' --tmax 7. "


		# train on trials that either share the shape or the color, test on all
			echo "python decoding.py --baseline -v 6 --clip --smooth 5 \
--crossval 'kfold' --n_folds 5 --train-cond 'one_object' --label ShareTrainObject \
-w --timegen -s $subject --sfreq 100 --test-cond 'two_objects' \
--train-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
--train-query-2 \"(Shape1=='$shape' and Colour1!='$colour') or (Shape1!='$shape' and Colour1=='$colour')\" \
--test-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
--test-query-2 \"\"Shape1!='$shape' or Colour1!='$colour'\" \
-i \"Data/Epochs2/\" -o \"/Epochs/\" --tmin '-0.5' --tmax 7. "

		# train on trials that either share the shape or the color and test on same trials
			echo "python decoding.py --baseline -v 6 --clip --smooth 5 \
--crossval 'kfold' --n_folds 5 --train-cond 'one_object' --label ShareTestObject \
-w --timegen -s $subject --sfreq 100 --test-cond 'two_objects' \
--train-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
--train-query-2 \"(Shape1=='$shape' and Colour1!='$colour') or (Shape1!='$shape' and Colour1=='$colour')\" \
--test-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
--test-query-2 \"(Shape1=='$shape' and Colour1!='$colour') or (Shape1!='$shape' and Colour1=='$colour')\" \
-i \"Data/Epochs2/\" -o \"/Epochs/\" --tmin '-0.5' --tmax 7. "



		# train on all trials and test on match trials only to get the same image, on first or second position
			echo "python decoding.py --baseline -v 6 --clip --smooth 5 \
--crossval 'kfold' --n_folds 5 --train-cond 'one_object' --label MatchObject \
-w --timegen -s $subject --sfreq 100 --test-cond 'two_objects' \
--train-query-1 \"(Shape1=='$shape') and (Colour1=='$colour') and (Matching=='match')\" \
--train-query-2 \"(Shape1!='$shape') or (Colour1!='$colour') and (Matching=='match')\" \
--test-query-1 \"(Shape1=='$shape' and Colour1=='$colour') or (Shape2=='$shape' and Colour2=='$colour') and (and Matching=='match')\" \
--test-query-2 \"(Shape1!='$shape' or Colour1!='$colour') and (Shape2!='$shape' or Colour2=='$colour') and (and Matching=='match')\" \
-i \"Data/Epochs2/\" -o \"/Epochs/\" --tmin '-0.5' --tmax 7. "

	done
done


# IMG LOCALIZER IMAGES
for colour in rouge bleu vert
do
	# train on imglocalizer images 
	echo "python decoding.py --baseline -v 6 --clip --smooth 5 \
--crossval 'kfold' --n_folds 5 --train-cond 'imgloc' --label ImgC \
--test-cond 'one_object' --test-cond 'two_objects' \
-w --timegen -s $subject --sfreq 100 \
--train-query-1 \"Loc_word=='img_$colour'\" \
--train-query-2 \"Loc_word!='img_$colour'\" \
--test-query-1 \"Colour1=='$colour'\" \
--test-query-2 \"Colour1!='$colour'\" \
-i \"Data/Epochs2/\" -o \"/Epochs/\" --tmin '-0.5' --tmax 7. "
done


for shape in carre cercle triangle
do
	# train on imglocalizer
	echo "python decoding.py --baseline -v 6 --clip --smooth 5 \
--crossval 'kfold' --n_folds 5 --train-cond 'imgloc' --label ImgS \
--test-cond 'one_object' --test-cond 'two_objects' \
-w --timegen -s $subject --sfreq 100 \
--train-query-1 \"Loc_word=='img_$shape'\" \
--train-query-2 \"Loc_word!='img_$shape'\" \
--test-query-1 \"Shape1=='$shape'\" \
--test-query-2 \"Shape1!='$shape'\" \
-i \"Data/Epochs2/\" -o \"/Epochs/\" --tmin '-0.5' --tmax 7. "
done


# IMG LOCALIZER IMAGES - TEST ON MATCH ONLY (SO THAT THE IMAGES MATCH)
for colour in rouge bleu vert
do
	# train on imglocalizer images 
	echo "python decoding.py --baseline -v 6 --clip --smooth 5 \
--crossval 'kfold' --n_folds 5 --train-cond 'imgloc' --label ImgC \
--test-cond 'one_object' --test-cond 'two_objects' \
-w --timegen -s $subject --sfreq 100 \
--train-query-1 \"Loc_word=='img_$colour'\" \
--train-query-2 \"Loc_word!='img_$colour'\" \
--test-query-1 \"Colour1=='$colour' and Matching=='match'\" \
--test-query-2 \"Colour1!='$colour' and Matching=='match'\" \
-i \"Data/Epochs2/\" -o \"/Epochs/\" --tmin '-0.5' --tmax 7. "
done

for shape in carre cercle triangle
do
	# train on imglocalizer
	echo "python decoding.py --baseline -v 6 --clip --smooth 5 \
--crossval 'kfold' --n_folds 5 --train-cond 'imgloc' --label ImgS \
--test-cond 'one_object' --test-cond 'two_objects' \
-w --timegen -s $subject --sfreq 100 \
--train-query-1 \"Loc_word=='img_$shape'\" \
--train-query-2 \"Loc_word!='img_$shape'\" \
--test-query-1 \"Shape1=='$shape' and Matching=='match'\" \
--test-query-2 \"Shape1!='$shape' and Matching=='match'\" \
-i \"Data/Epochs2/\" -o \"/Epochs/\" --tmin '-0.5' --tmax 7. "
done



# --localizer --path2loc 'Single_Chan_vs15/CMR_clean_sent' --pval-thresh 0.01 \
	# done
done