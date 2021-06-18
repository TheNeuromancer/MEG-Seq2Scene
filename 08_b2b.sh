for sub in $(python configs/config.py)
do

## JUST OBJECTS
	echo "python 08_b2b.py -w \
--timegen -s $sub --label Obj \
--queries \"Shape1\" \
--train_cond 'one_object' \
--queries \"Colour1\" \
--queries \"Shape1+Colour1\" "

## JUST SCENES
	echo "python 08_b2b.py -w \
--timegen -s $sub --label Scenes \
--train_cond 'two_objects' \
--queries \"Shape1\" \
--queries \"Colour1\" \
--queries \"Shape2\" \
--queries \"Colour2\" \
--queries \"Shape1+Colour1\" \
--queries \"Shape2+Colour2\" "


## ALL TOGETHER
	echo "python 08_b2b.py -w \
--timegen -s $sub --label AllTogether \
--train_cond 'localizer' \
--train_cond 'one_object' \
--train_cond 'two_objects' \
--queries \"Shape1\" \
--queries \"Colour1\" \
--queries \"Shape2\" \
--queries \"Colour2\" \
--queries \"Shape1+Colour1\" \
--queries \"Shape2+Colour2\" "


done