cd ${SPARK_HOME}

export MASTER="spark://$SPARK_MASTER_NODE:$SPARK_MASTER_PORT"

export PYSPARK_DRIVER_PYTHON="/lustre/home/rceccaroni/.conda/envs/python/bin/python"
export PYSPARK_PYTHON="/lustre/home/rceccaroni/.conda/envs/python/bin/python"
export PYTHONPATH="/lustre/home/rceccaroni/.conda/envs/python/bin/python"
export PYSPARK_DRIVER_PYTHON="jupyter"
export PYSPARK_DRIVER_PYTHON_OPTS="notebook --notebook-dir=../../ --no-browser --port=9999"


command="bin/pyspark --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=/lustre/home/rceccaroni/.conda/envs/python/bin/python"

echo "Running $command" >&2
$command