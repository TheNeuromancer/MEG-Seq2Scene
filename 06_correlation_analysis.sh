for sub in $(python configs/config.py)
do
	echo "python 06_correlation_analysis.py -s $sub -w --windows '3,5'"
	echo "python 06_correlation_analysis.py -s $sub -w --mirror_img  --windows '3,5'"

	echo "python 06_correlation_analysis.py -s $sub -w"
	echo "python 06_correlation_analysis.py -s $sub -w --mirror_img"
done

