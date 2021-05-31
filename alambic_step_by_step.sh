#!/bin/sh

##############################################
### EXAMPLE TO RUN MANY JOBS WITH QSUB     ###
### CREATE THE qsub FILES INTO A LOOP      ###
##############################################
#
# Arguments:
# $1: file containing all the commands, one per line
#

# Warn and exit if there is not input file
if [ -z "$1" ]; then
	echo 'Expecting an input file containing the commands to run... exiting'
	exit
fi

max_running_jobs=700 # max nb of jobs running at the same time

# load commands from file
IFS=$'\r\n' GLOBIGNORE='*' command eval  'job_array=($(cat $1))'

# set up paths 
timestamp=$(date "+%Y.%m.%d-%H.%M.%S")
alambic_logs_path='/neurospin/unicog/protocols/MEG/Seq2Scene/Alambic_logs/'$timestamp
alambic_job_files_path='/neurospin/unicog/protocols/MEG/Seq2Scene/Alambic_job_files/'$timestamp
mkdir $alambic_logs_path
mkdir $alambic_job_files_path

for i in `seq 0 ${#job_array[@]}`;
do
  file_qsub=$alambic_job_files_path/file_qsub_$i.qs
  out_file=$alambic_logs_path/out_$i
  err_file=$alambic_logs_path/err_$i

  # get number of running jobs and compared to maximum authorized
  while [ $(qselect -u td260249 | wc -l) -ge $max_running_jobs ]
  do
  	sleep 1
  done

cat <<EOT >> $file_qsub
#!/bin/bash
#PBS -N $i
#PBS -l walltime=29:59:00
#PBS -l ncpus=1
#PBS -l mem=5G
#PBS -q Nspin_long
#PBS -o $out_file
#PBS -e $err_file
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neurospin
cd /neurospin/unicog/protocols/MEG/Seq2Scene/Code/
${job_array[i]}
EOT

qsub $file_qsub

sleep 1 # time to let the job start, with margin

done

#PBS -l nodes=1:ppn=8
