for sub in $(python configs/config.py) # all
do
	# for freq_band in low_high low_low high_low high_vlow low_vlow low_vhigh high_vhigh vhigh_vhigh vhigh_high vhigh_low
	# do

# ## OBJECTS
# 	echo "python 04_decoding_window.py -w \
# -s $sub --label AllObjectsfull \
# --train-cond 'two_objects' \
# --train-query \"Shape1+Colour1\" \
# --windows \"0.,1.4\" \
# --train-cond 'one_object' \
# --train-query \"Shape1+Colour1\" \
# --windows \"0.,1.4\" \
# --train-cond 'two_objects' \
# --train-query \"Shape2+Colour2\" \
# --windows \"1.8,3.2\" "



# ##### COLORS #####
	## FULL
	echo "python 04_decoding_window.py -w \
-s $sub --label Colourfull \
--train-cond 'two_objects' \
--train-query \"Colour1\" \
--windows \"0.7,1.3\" \
--train-cond 'one_object' \
--train-query \"Colour1\" \
--windows \"0.7,1.3\" \
--train-cond 'two_objects' \
--train-query \"Colour2\" \
--windows \"2.5,3.1\" "

# # 	## EARLY
# 	echo "python 04_decoding_window.py -w \
# -s $sub --label Colourearly \
# --train-cond 'two_objects' \
# --train-query \"Colour1\" \
# --windows \"0.7,1.\" \
# --train-cond 'one_object' \
# --train-query \"Colour1\" \
# --windows \"0.7,1.\" \
# --train-cond 'two_objects' \
# --train-query \"Colour2\" \
# --windows \"2.5, 2.8\" "

# 	## LATE
# 	echo "python 04_decoding_window.py -w \
# -s $sub --label Colourlate \
# --train-cond 'two_objects' \
# --train-query \"Colour1\" \
# --windows \"1.05,1.35\" \
# --train-cond 'one_object' \
# --train-query \"Colour1\" \
# --windows \"1.05,1.35\" \
# --train-cond 'two_objects' \
# --train-query \"Colour2\" \
# --windows \"2.85, 3.15\" "


# ##### SHAPES #####
	## FULL
	echo "python 04_decoding_window.py -w \
-s $sub --label Shapefull \
--train-cond 'two_objects' \
--train-query \"Shape1\" \
--windows \"0.1,.7\" \
--train-cond 'one_object' \
--train-query \"Shape1\" \
--windows \"0.1,.7\" \
--train-cond 'two_objects' \
--train-query \"Shape2\" \
--windows \"1.9,2.5\" "

# 	## EARLY
# 	echo "python 04_decoding_window.py -w \
# -s $sub --label Shapeearly \
# --train-cond 'two_objects' \
# --train-query \"Shape1\" \
# --windows \".1,.4\" \
# --train-cond 'one_object' \
# --train-query \"Shape1\" \
# --windows \".1,.4\" \
# --train-cond 'two_objects' \
# --train-query \"Shape2\" \
# --windows \"1.9,2.2\" "

# ## LATE
# 	echo "python 04_decoding_window.py -w \
# -s $sub --label Shapelate \
# --train-cond 'two_objects' \
# --train-query \"Shape1\" \
# --windows \".45,.75\" \
# --train-cond 'one_object' \
# --train-query \"Shape1\" \
# --windows \".45,.75\" \
# --train-cond 'two_objects' \
# --train-query \"Shape2\" \
# --windows \"2.25,2.55\" "


#################################################

## During the DELAY
## OBJECTS
# 	echo "python 04_decoding_window.py -w \
# -s $sub --label Objectsdelay \
# --train-cond 'two_objects' \
# --train-query \"Shape1+Colour1\" \
# --windows \"4.,4.6\" \
# --train-cond 'two_objects' \
# --train-query \"Shape2+Colour2\" \
# --windows \"4.,4.6\" \
# --train-cond 'one_object' \
# --train-query \"Shape1+Colour1\" \
# --windows \"1.5,2.1\" "
# # --windows \"3.5,5.\" "

## SHAPES
	echo "python 04_decoding_window.py -w \
-s $sub --label Shapedelay \
--train-cond 'two_objects' \
--train-query \"Shape1\" \
--windows \"4.,4.6\" \
--train-cond 'two_objects' \
--train-query \"Shape2\" \
--windows \"4.,4.6\" \
--train-cond 'one_object' \
--train-query \"Shape1\" \
--windows \"1.5,2.1\" "

## COLORS
	echo "python 04_decoding_window.py -w \
-s $sub --label Colourdelay \
--train-cond 'two_objects' \
--train-query \"Colour1\" \
--windows \"4.,4.6\" \
--train-cond 'two_objects' \
--train-query \"Colour2\" \
--windows \"4.,4.6\" \
--train-cond 'one_object' \
--train-query \"Colour1\" \
--windows \"1.5,2.1\" "


## SIDE SHAPES
	echo "python 04_decoding_window.py -w \
-s $sub --label SideShape_delay \
--train-cond 'two_objects' \
--train-query \"Left_shape\" \
--windows \"4.,4.6\" \
--train-cond 'two_objects' \
--train-query \"Right_shape\" \
--windows \"4.,4.6\" "

## SIDE COLORS
	echo "python 04_decoding_window.py -w \
-s $sub --label SideColour_delay \
--train-cond 'two_objects' \
--train-query \"Left_color\" \
--windows \"4.,4.6\" \
--train-cond 'two_objects' \
--train-query \"Right_color\" \
--windows \"4.,4.6\" "
# 	echo "python 04_decoding_window.py -w \
# -s $sub --label SideObjectsdelay \
# --train-cond 'two_objects' \
# --train-query \"Left_obj\" \
# --windows \"4.,4.6\" \
# --train-cond 'two_objects' \
# --train-query \"Right_obj\" \
# --windows \"4.,4.6\" \
# --train-cond 'one_object' \
# --train-query \"Shape1+Colour1\" \
# --windows \"1.5,2.1\" "


done
