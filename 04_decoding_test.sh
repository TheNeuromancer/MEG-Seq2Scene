# for quality testing. Does decoding on individual run_nbs to get a measure of data quality. 

for sub in $(python configs/config.py)
do

	for run_nb in 1 2 3 4 5 6 7 8 9 10
	do

	# train on one object
	echo "python 04_decoding_ovr.py -w --test_quality \
--timegen -s $sub --label Colour \
--train-cond 'one_object' \
--train-query \"Colour1\" \
--filter \"run_nb=='$run_nb'\" "

	# train on two objects
	echo "python 04_decoding_ovr.py -w --test_quality \
--timegen -s $sub --label Colour1 \
--train-cond 'two_objects' \
--train-query \"Colour1\" \
--filter \"run_nb=='$run_nb'\" "

	echo "python 04_decoding_ovr.py -w --test_quality \
--timegen -s $sub --label Colour2 \
--train-cond 'two_objects' \
--train-query \"Colour2\" \
--filter \"run_nb=='$run_nb'\" "

## SHAPE
# 	# train on localizer words only
	echo "python 04_decoding_ovr.py -w --test_quality --train-cond 'localizer' --label Shape \
--timegen -s $sub \
--train-query \"Loc_word\" \
--filter \"run_nb=='$run_nb'\" "

	# train on one object
	echo "python 04_decoding_ovr.py -w --test_quality \
--timegen -s $sub --label Shape \
--train-cond 'one_object' \
--train-query \"Shape1\" \
--filter \"run_nb=='$run_nb'\" "

	# train on two objects
	echo "python 04_decoding_ovr.py -w --test_quality \
--timegen -s $sub --label Shape1 \
--train-cond 'two_objects' \
--train-query \"Shape1\" \
--filter \"run_nb=='$run_nb'\" "

	echo "python 04_decoding_ovr.py -w --test_quality \
--timegen -s $sub --label Shape2 \
--train-cond 'two_objects' \
--train-query \"Shape2\" \
--filter \"run_nb=='$run_nb'\" "


## OBJECTS
		# train on all other trials, gen to first, then to 2nd object
	echo "python 04_decoding_ovr.py -w --test_quality \
--timegen -s $sub --train-cond 'one_object' --label AllObject \
--train-query \"Shape1+Colour1\" \
--filter \"run_nb=='$run_nb'\" "

		# train on scenes 1st obj
	echo "python 04_decoding_ovr.py -w --test_quality \
--timegen -s $sub --train-cond 'two_objects' --label All1stObj \
--train-query \"Shape1+Colour1\" \
--filter \"run_nb=='$run_nb'\" "

		# train on scenes 2nd obj
	echo "python 04_decoding_ovr.py -w --test_quality \
--timegen -s $sub --train-cond 'two_objects' --label All2ndObj \
--train-query \"Shape2+Colour2\" \
--filter \"run_nb=='$run_nb'\" "


## TRAIN OBJECTS ON SCENES (ORDER OF THE VISUAL SCENE)

		# RIGHT OBJECT
	echo "python 04_decoding_ovr.py -w --test_quality \
--label AllRightObject --timegen -s $sub \
--train-cond 'two_objects' \
--train-query \"Right_obj\" \
--filter \"run_nb=='$run_nb'\" "
	
		# LEFT OBJECT
	echo "python 04_decoding_ovr.py -w --test_quality \
--label AllLeftObject --timegen -s $sub \
--train-cond 'two_objects' \
--train-query \"Left_obj\" \
--filter \"run_nb=='$run_nb'\" "


## RELATION
	echo "python 04_decoding_ovr.py -w --test_quality -c v1 \
--timegen -s $sub --label Relation \
--train-cond 'two_objects' \
--train-query \"Relation\" \
--filter \"run_nb=='$run_nb'\" "



## USE WINDOWS AROUND EACH COND

	## COLOUR
	# train on one object
	echo "python 04_decoding_ovr.py -w --test_quality \
--timegen -s $sub --label winColour \
--train-cond 'one_object' \
--windows \"0.6,1.4\" \
--train-query \"Colour1\" \
--filter \"run_nb=='$run_nb'\" "

	# train on two objects
	echo "python 04_decoding_ovr.py -w --test_quality \
--timegen -s $sub --label winColour1 \
--train-cond 'two_objects' \
--windows \"0.6,1.4\" \
--train-query \"Colour1\" \
--filter \"run_nb=='$run_nb'\" "

	echo "python 04_decoding_ovr.py -w --test_quality \
--timegen -s $sub --label winColour2 \
--train-cond 'two_objects' \
--windows \"2.4, 3.2\" \
--train-query \"Colour2\" \
--filter \"run_nb=='$run_nb'\" "


	## SHAPES
	# train on one object
	echo "python 04_decoding_ovr.py -w --test_quality \
--timegen -s $sub --label winShape \
--train-cond 'one_object' \
--windows \"0.,.8\" \
--train-query \"Shape1\" \
--filter \"run_nb=='$run_nb'\" "

	# train on two objects
	echo "python 04_decoding_ovr.py -w --test_quality \
--timegen -s $sub --label winShape1 \
--train-cond 'two_objects' \
--windows \"0.,.8\" \
--train-query \"Shape1\" \
--filter \"run_nb=='$run_nb'\" "

	echo "python 04_decoding_ovr.py -w --test_quality \
--timegen -s $sub --label winShape2 \
--train-cond 'two_objects' \
--windows \"1.8,2.6\" \
--train-query \"Shape2\" \
--filter \"run_nb=='$run_nb'\" "


	## OBJECTS
	# train on all other trials, gen to first, then to 2nd object
	echo "python 04_decoding_ovr.py -w --test_quality \
--timegen -s $sub --label winAllObject \
--train-cond 'one_object' \
--windows \"0.,1.4\" \
--train-query \"Shape1+Colour1\" \
--filter \"run_nb=='$run_nb'\" "

	# train on scenes 1st obj
	echo "python 04_decoding_ovr.py -w --test_quality \
--timegen -s $sub --label winAll1stObj \
--train-cond 'two_objects' \
--windows \"0.,1.4\" \
--train-query \"Shape1+Colour1\" \
--filter \"run_nb=='$run_nb'\" "

	# train on scenes 2nd obj
	echo "python 04_decoding_ovr.py -w --test_quality \
--timegen -s $sub --label winAll2ndObj \
--train-cond 'two_objects' \
--windows \"1.8,3.2\" \
--train-query \"Shape2+Colour2\" \
--filter \"run_nb=='$run_nb'\" "

done
done