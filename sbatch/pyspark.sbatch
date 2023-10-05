#!/bin/sh
#############################################################################
#  Copyright (C) 2013-2015 Lawrence Livermore National Security, LLC.
#  Produced at Lawrence Livermore National Laboratory (cf, DISCLAIMER).
#  Written by Albert Chu <chu11@llnl.gov>
#  LLNL-CODE-644248
#
#  This file is part of Magpie, scripts for running Hadoop on
#  traditional HPC systems.  For details, see https://github.com/llnl/magpie.
#
#  Magpie is free software; you can redistribute it and/or modify it
#  under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  Magpie is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#  General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with Magpie.  If not, see <http://www.gnu.org/licenses/>.
#############################################################################

############################################################################
# SLURM Customizations
############################################################################

# Node count.  Node count should include one node for the
# head/management/master node.  For example, if you want 8 compute
# nodes to process data, specify 9 nodes below.
#
# If including Zookeeper, include expected Zookeeper nodes.  For
# example, if you want 8 Hadoop compute nodes and 3 Zookeeper nodes,
# specify 12 nodes (1 master, 8 Hadoop, 3 Zookeeper)
#
# Also take into account additional nodes needed for other services.
#
# Many of the below can be configured on the command line.  If you are
# more comfortable specifying these on the command line, feel free to
# delete the customizations below.

#SBATCH --nodes=4
#SBATCH --nodelist=cn4d,cn5a,cn5c,cn5d
#SBATCH --output="slurm-%j.out"

# Note defaults of MAGPIE_STARTUP_TIME & MAGPIE_SHUTDOWN_TIME, this
# timelimit should be a fair amount larger than them combined.
#SBATCH --time=05:00:00

# Job name.  This will be used in naming directories for the job.
#SBATCH --job-name=spark

# Partition to launch job in
#SBATCH --partition=dss

## SLURM Values
# Generally speaking, don't touch the following, misc other configuration

#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --no-kill

#SBATCH --mem=64G

module load jdk/8

# Need to tell Magpie how you are submitting this job
export MAGPIE_SUBMISSION_TYPE="sbatchsrun"

export PYSPARK_DRIVER_PYTHON="/lustre/home/rceccaroni/.conda/envs/python/bin/python"
export PYSPARK_PYTHON="/lustre/home/rceccaroni/.conda/envs/python/bin/python"
export PYTHONPATH="/lustre/home/rceccaroni/.conda/envs/python/bin/python"

############################################################################
# Magpie Configurations
############################################################################

export MAGPIE_SCRIPTS_HOME="${HOME}/magpie"
export MAGPIE_NO_LOCAL_DIR="yes"
export MAGPIE_LOCAL_DIR="/lustre/home/${USER}/tmp/magpie"

# Magpie job type
export MAGPIE_JOB_TYPE="script"

THIS_SCRIPT=$(readlink -f "$0")
THIS_PATH=$(dirname "THIS_SCRIPT")

export MAGPIE_JOB_SCRIPT="$(dirname "$THIS_PATH")/scripts/pyspark.sh"

# Specify script startup / shutdown time window
export MAGPIE_STARTUP_TIME=30
export MAGPIE_SHUTDOWN_TIME=30

# Hostname config
export MAGPIE_HOSTNAME_CMD="hostname -s"

# Map Hostnames
export MAGPIE_HOSTNAME_CMD_MAP="${MAGPIE_SCRIPTS_HOME}/scripts/hostname-map-scripts/short_name.sh"
#
export MAGPIE_HOSTNAME_SCHEDULER_MAP="${MAGPIE_SCRIPTS_HOME}/scripts/hostname-map-scripts/short_name.sh"

############################################################################
# General Configuration
############################################################################
export MAGPIE_PYTHON="/lustre/home/rceccaroni/.conda/envs/python/bin/python"

############################################################################
# Hadoop Core Configurations
############################################################################
export HADOOP_SETUP=yes
export HADOOP_SETUP_TYPE="HDFS"
export HADOOP_VERSION="3.3.0"
export HADOOP_HOME="${HOME}/bigdata/hadoop-${HADOOP_VERSION}"
export HADOOP_LOCAL_DIR="/lustre/home/${USER}/tmp/hadoop"
export HADOOP_CONF_FILES="${HOME}/myscripts/conf/hadoop"
export HADOOP_FILESYSTEM_MODE="hdfsoverlustre"
export HADOOP_HDFS_REPLICATION=3
export HADOOP_HDFSOVERLUSTRE_PATH="/lustre/home/${USER}/hdfsoverlustre/"

############################################################################
# Spark Core Configurations
############################################################################
export SPARK_SETUP=yes
export SPARK_SETUP_TYPE="STANDALONE"
export SPARK_VERSION="3.3.0-bin-hadoop3"
export SPARK_HOME="${HOME}/bigdata/spark-${SPARK_VERSION}"
export SPARK_LOCAL_DIR="/lustre/home/${USER}/tmp/spark"
# export SPARK_CONF_FILES="${HOME}/myconf"
export SPARK_WORKER_CORES_PER_NODE=32
export SPARK_WORKER_MEMORY_PER_NODE=64000
# export SPARK_WORKER_DIRECTORY=/lustre/home/${USER}/tmp/spark/work
export SPARK_JOB_MEMORY=${SPARK_WORKER_MEMORY_PER_NODE}
export SPARK_DRIVER_MEMORY=${SPARK_WORKER_MEMORY_PER_NODE}
export SPARK_DAEMON_HEAP_MAX=2000
# SPARK_DEFAULT_PARALLELISM
export SPARK_DEFAULT_PARALLELISM=32

# SPARK_MEMORY_FRACTION
export SPARK_MEMORY_FRACTION=0.99

# SPARK_MEMORY_STORAGE_FRACTION
# export SPARK_MEMORY_STORAGE_FRACTION=0.5

# SPARK_STORAGE_MEMORY_FRACTION
# export SPARK_STORAGE_MEMORY_FRACTION=0.6

# SPARK_SHUFFLE_MEMORY_FRACTION
# export SPARK_SHUFFLE_MEMORY_FRACTION=0.2

############################################################################
# Run Job
############################################################################

srun --no-kill -W 0 $MAGPIE_SCRIPTS_HOME/magpie-check-inputs
if [ $? -ne 0 ]
then
    exit 1
fi
srun --no-kill -W 0 $MAGPIE_SCRIPTS_HOME/magpie-setup-core
if [ $? -ne 0 ]
then
    exit 1
fi
srun --no-kill -W 0 $MAGPIE_SCRIPTS_HOME/magpie-setup-projects
if [ $? -ne 0 ]
then
    exit 1
fi
srun --no-kill -W 0 $MAGPIE_SCRIPTS_HOME/magpie-setup-post
if [ $? -ne 0 ]
then
    exit 1
fi
srun --no-kill -W 0 $MAGPIE_SCRIPTS_HOME/magpie-pre-run
if [ $? -ne 0 ]
then
    exit 1
fi
srun --no-kill -W 0 $MAGPIE_SCRIPTS_HOME/magpie-run
srun --no-kill -W 0 $MAGPIE_SCRIPTS_HOME/magpie-cleanup
srun --no-kill -W 0 $MAGPIE_SCRIPTS_HOME/magpie-post-run
