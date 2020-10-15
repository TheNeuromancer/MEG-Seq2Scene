for subject in theo
do
	# for freq_band in low_high low_low high_low high_vlow low_vlow low_vhigh high_vhigh vhigh_vhigh vhigh_high vhigh_low
	# do

for colour in rouge bleu vert
do
	# train on localizer
	echo "python decoding.py --baseline -v 3 --clip --smooth 5 \
--crossval 'kfold' --n_folds 5 --train-cond 'localizer' \
--test-cond 'one_object' --test-cond 'two_objects' \
-w --timegen -s $subject --sfreq 50 \
--train-query-1 \"Loc_word=='$colour'\" \
--train-query-2 \"Loc_word!='$colour'\" \
--test-query-1 \"Colour1=='$colour'\" \
--test-query-2 \"Colour1!='$colour'\" \
-i \"Data/Epochs2/\" -o \"/Epochs/\" --tmin '-0.2' --tmax 6. "

	# train on imglocalizer
	echo "python decoding.py --baseline -v 3 --clip --smooth 5 \
--crossval 'kfold' --n_folds 5 --train-cond 'imgloc' \
--test-cond 'one_object' --test-cond 'two_objects' \
-w --timegen -s $subject --sfreq 50 \
--train-query-1 \"Loc_word=='$colour'\" \
--train-query-2 \"Loc_word!='$colour'\" \
--test-query-1 \"Colour1=='$colour'\" \
--test-query-2 \"Colour1!='$colour'\" \
-i \"Data/Epochs2/\" -o \"/Epochs/\" --tmin '-0.2' --tmax 6. "

	# train on one object
	echo "python decoding.py --baseline -v 3 --clip --smooth 5 \
--crossval 'kfold' --n_folds 5 --train-cond 'one_object' \
-w --timegen -s $subject --sfreq 50 --test-cond 'two_objects' \
--train-query-1 \"Colour1=='$colour'\" \
--train-query-2 \"Colour1!='$colour'\" \
--test-query-1 \"Colour1=='$colour'\" \
--test-query-2 \"Colour1!='$colour'\" \
-i \"Data/Epochs2/\" -o \"/Epochs/\" --tmin '-0.5' --tmax 6. "
	# test on 2nd colour
	echo "python decoding.py --baseline -v 3 --clip --smooth 5 \
--crossval 'kfold' --n_folds 5 --train-cond 'one_object' \
-w --timegen -s $subject --sfreq 50 --test-cond 'two_objects' \
--train-query-1 \"Colour1=='$colour'\" \
--train-query-2 \"Colour1!='$colour'\" \
--test-query-1 \"Colour2=='$colour'\" \
--test-query-2 \"Colour2!='$colour'\" \
-i \"Data/Epochs2/\" -o \"/Epochs/\" --tmin '-0.5' --tmax 6. "
done


for shape in carre cercle triangle
do
	# train on localizer
	echo "python decoding.py --baseline -v 3 --clip --smooth 5 \
--crossval 'kfold' --n_folds 5 --train-cond 'localizer' \
--test-cond 'one_object' --test-cond 'two_objects' \
-w --timegen -s $subject --sfreq 50 \
--train-query-1 \"Loc_word=='$shape'\" \
--train-query-2 \"Loc_word!='$shape'\" \
--test-query-1 \"Shape1=='$shape'\" \
--test-query-2 \"Shape1!='$shape'\" \
-i \"Data/Epochs2/\" -o \"/Epochs/\" --tmin '-0.2' --tmax 6. "

	# train on imglocalizer
	echo "python decoding.py --baseline -v 3 --clip --smooth 5 \
--crossval 'kfold' --n_folds 5 --train-cond 'imgloc' \
--test-cond 'one_object' --test-cond 'two_objects' \
-w --timegen -s $subject --sfreq 50 \
--train-query-1 \"Loc_word=='$shape'\" \
--train-query-2 \"Loc_word!='$shape'\" \
--test-query-1 \"Shape1=='$shape'\" \
--test-query-2 \"Shape1!='$shape'\" \
-i \"Data/Epochs2/\" -o \"/Epochs/\" --tmin '-0.2' --tmax 6. "

	# train on one object
	echo "python decoding.py --baseline -v 3 --clip --smooth 5 \
--crossval 'kfold' --n_folds 5 --train-cond 'one_object' \
-w --timegen -s $subject --sfreq 50 --test-cond 'two_objects' \
--train-query-1 \"Shape1=='$shape'\" \
--train-query-2 \"Shape1!='$shape'\" \
--test-query-1 \"Shape1=='$shape'\" \
--test-query-2 \"Shape1!='$shape'\" \
-i \"Data/Epochs2/\" -o \"/Epochs/\" --tmin '-0.5' --tmax 6. "
	# test on 2nd shape
	echo "python decoding.py --baseline -v 3 --clip --smooth 5 \
--crossval 'kfold' --n_folds 5 --train-cond 'one_object' \
-w --timegen -s $subject --sfreq 50 --test-cond 'two_objects' \
--train-query-1 \"Shape1=='$shape'\" \
--train-query-2 \"Shape1!='$shape'\" \
--test-query-1 \"Shape2=='$shape'\" \
--test-query-2 \"Shape2!='$shape'\" \
-i \"Data/Epochs2/\" -o \"/Epochs/\" --tmin '-0.5' --tmax 6. "
done



## TRAIN OBJECTS
for colour in rouge bleu vert
do
	for shape in carre cercle triangle
	do
			echo "python decoding.py --baseline -v 3 --clip --smooth 5 \
--crossval 'kfold' --n_folds 5 --train-cond 'one_object' \
-w --timegen -s $subject --sfreq 50 --test-cond 'two_objects' \
--train-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
--train-query-2 \"Shape1!='$shape' and Colour1!='$colour'\" \
--test-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
--test-query-2 \"Shape1!='$shape' and Colour1!='$colour'\" \
-i \"Data/Epochs2/\" -o \"/Epochs/\" --tmin '-0.5' --tmax 6. "
			# test on 2nd shape
			echo "python decoding.py --baseline -v 3 --clip --smooth 5 \
--crossval 'kfold' --n_folds 5 --train-cond 'one_object' \
-w --timegen -s $subject --sfreq 50 --test-cond 'two_objects' \
--train-query-1 \"Shape1=='$shape' and Colour1=='$colour'\" \
--train-query-2 \"Shape1!='$shape' and Colour1!='$colour'\" \
--test-query-1 \"Shape2=='$shape' and Colour2=='$colour'\" \
--test-query-2 \"Shape2!='$shape' and Colour2!='$colour'\" \
-i \"Data/Epochs2/\" -o \"/Epochs/\" --tmin '-0.5' --tmax 6. "
	done
done


# --localizer --path2loc 'Single_Chan_vs15/CMR_clean_sent' --pval-thresh 0.01 \
	# done
done