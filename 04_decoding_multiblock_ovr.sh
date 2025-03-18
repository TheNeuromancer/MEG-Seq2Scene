for sub in $(python configs/config.py)
do

### SINGLE TIME POINT TRAINING (gen to the delay)
## addition function
add() { n="$@"; bc <<< "${n// /+}"; }

# for t in 0.17 0.2 0.3 0.4 0.5 0.6 0.8
# for t in 0.2 0.4 0.6 0.8
for t in 0.18 0.26 0.36
do
	c1t=$(add $t 0.6)
	rt=$(add $t 1.2)
	s2t=$(add $t 1.8)
	c2t=$(add $t 2.4)

echo "python 04_decoding_multiblock_ovr.py -w -s $sub --label CrossAll$t \
--train-conds 'localizer' \
--train-query 'Loc_crossAll' --windows '$t,$t' \
--test-cond 'two_objects' \
--test-query 'Property' --windows '3, 5' "

# echo "python 04_decoding_multiblock_ovr.py -w -s $sub --null_prop 0.5 --split_props --label Prop$t \
# --train-conds 'two_objects' \
# --train-query 'Property' --windows '$t,$t' \
# --test-cond 'two_objects' \
# --test-query 'Property' --windows '3, 5' "


# # echo "python 04_decoding_multiblock_ovr.py -w -s $sub --null_prop 0.5 --split_props --label Prop$t \
# # --train-conds 'two_objects' --train-conds 'localizer' \
# # --train-query 'Property' --windows '$t,$t' \
# # --test-cond 'two_objects' \
# # --test-query 'Property' --windows '3, 5' "


# echo "python 04_decoding_multiblock_ovr.py -w -s $sub --null_prop 0.5 --split_props --label Prop$t \
# --train-conds 'two_objects' --train-conds 'localizer' --train-conds 'one_object' \
# --train-query 'Property' --windows '$t,$t' \
# --test-cond 'two_objects' \
# --test-query 'Property' --windows '3, 5' "

# # --train-conds 'one_object' 
# # --test-cond 'one_object' \
# # --test-query 'Property' --windows '1.5, 2.2' \


# echo "python 04_decoding_multiblock_ovr.py -w -s $sub --null_prop 0.5 --split_props --label PropAll$t \
# --train-conds 'two_objects' \
# --train-query 'PropertyAll' --windows '$t,$t' \
# --test-cond 'two_objects' \
# --test-query 'PropertyAll' --windows '3, 5' "

# echo "python 04_decoding_multiblock_ovr.py -w -s $sub --null_prop 0.5 --split_props --label PropAll$t \
# --train-conds 'one_object' --train-conds 'localizer' \
# --train-query 'PropertyAll' --windows '$t,$t' \
# --test-cond 'two_objects' \
# --test-query 'PropertyAll' --windows '3, 5' "

# echo "python 04_decoding_multiblock_ovr.py -w -s $sub --split_props --null_prop 0.1 --label Prop$t \
# --train-conds 'two_objects' --train-conds 'localizer' --train-conds 'one_object' \
# --train-query 'Property' --windows '$t,$t' \
# --test-cond 'two_objects' \
# --test-query 'Property' --windows '3, 5' "

# echo "python 04_decoding_multiblock_ovr.py -w -s $sub --split_props --null_prop 0.1 --label PropAll$t \
# --train-conds 'two_objects' --train-conds 'localizer' --train-conds 'one_object' \
# --train-query 'PropertyAll' --windows '$t,$t' \
# --test-cond 'two_objects' \
# --test-query 'PropertyAll' --windows '3, 5' "

# echo "python 04_decoding_multiblock_ovr.py -w -s $sub --null_prop 0.1 --label WordPos$t \
# --train-conds 'two_objects' \
# --train-query 'WordPos' --windows '$t,$t' \
# --test-cond 'two_objects' \
# --test-query 'WordPos' --windows '3, 5' "

done
done
