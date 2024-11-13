#!/bin/sh

##############################################
### EXAMPLE TO RUN MANY JOBS WITH QSUB     ###
### CREATE THE qsub FILES INTO A LOOP      ###
##############################################
#
# Arguments:
# $1: file containing all the commands, one per line
#

# Warn and exit if there is not input file
if [ -z "$1" ]; then
	echo 'Expecting an input file containing the commands to run... exiting'
	exit
fi

max_running_jobs=100 # max nb of jobs running at the same time

# load commands from file
IFS=$'\r\n' GLOBIGNORE='*' command eval  'job_array=($(cat $1))'

# set up paths 
timestamp=$(date "+%Y.%m.%d-%H.%M.%S")
cluster_logs_path='/home/users/d/desborde/Documents/s2s/cluster_logs/'$timestamp
cluster_job_files_path='/home/users/d/desborde/Documents/s2s/cluster_job_files/'$timestamp
mkdir $cluster_logs_path
mkdir $cluster_job_files_path

for i in `seq 0 ${#job_array[@]}`;
do
  file_sbatch=$cluster_job_files_path/file_sbatch_$i.qs
  out_file=$cluster_logs_path/out_$i
  err_file=$cluster_logs_path/err_$i

  # # get number of running jobs and compared to maximum authorized
  # while [ $(qselect -u desborde | wc -l) -ge $max_running_jobs ]
  # do
  # 	sleep 1
  # done

cat <<EOT >> $file_sbatch
#!/bin/bash
#SBATCH --job-name=$i
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --partition=shared-cpu
#SBATCH --output=$out_file
#SBATCH --error=$err_file
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mne
cd ~/Documents/s2s/MEG-Seq2Scene/
${job_array[i]}
EOT

sbatch $file_sbatch

sleep .1 # time to let the job start, with margin

done
