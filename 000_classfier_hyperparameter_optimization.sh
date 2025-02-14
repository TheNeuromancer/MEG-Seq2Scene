for sub in $(python configs/config.py)
do
	echo "python 000_classfier_hyperparameter_optimization.py -w -s $sub"
done