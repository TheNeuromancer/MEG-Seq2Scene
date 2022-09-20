for sub in $(python configs/config.py)
do
		echo "python 04_decoding_regression.py -w \
-s $sub --label Complexity --timegen -c v25_config \
--train-cond 'two_objects' \
--train-query \"Complexity\" "
# --split-queries \"Matching=='match'\" \
# --split-queries \"Matching=='nonmatch'\" \
# --split-queries \"Flash=='0'\" \
# --split-queries \"Flash=='1'\" \
done