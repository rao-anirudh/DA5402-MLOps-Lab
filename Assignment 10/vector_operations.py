from pyspark.sql import SparkSession
import numpy as np
import time
import os

# Spark setup
spark = SparkSession.builder.appName("VectorBenchmark").getOrCreate()
sc = spark.sparkContext

# Benchmark settings
array_sizes = [100_000, 1_000_000, 3_000_000]
num_slices_list = [sc.defaultParallelism, os.cpu_count(), 100]
results = []


def benchmark_task(func):
    start = time.time()
    result = func()
    end = time.time()
    return end - start, result


for size in array_sizes:
    v1 = np.random.rand(size)
    v2 = np.random.rand(size)
    k = 3.14

    for num_slices in num_slices_list:
        # Create distributed arrays
        rdd1 = sc.parallelize(v1, numSlices=num_slices)
        rdd2 = sc.parallelize(v2, numSlices=num_slices)

        # Run tasks and record times
        dot_time, _ = benchmark_task(lambda: rdd1.zip(rdd2).map(lambda x: x[0] * x[1]).sum())
        scale_time, _ = benchmark_task(lambda: rdd1.map(lambda x: k * x).count())
        add_time, _ = benchmark_task(lambda: rdd1.zip(rdd2).map(lambda x: x[0] + x[1]).count())

        total_time = dot_time + scale_time + add_time

        results.append({
            "size": size,
            "slices": num_slices,
            "dot_time": dot_time,
            "scale_time": scale_time,
            "add_time": add_time,
            "total_time": total_time
        })

header = ['Size', 'Slices', 'Total Time']

# Print the header
print(','.join(header))

# Print the data rows in CSV format with total time
for r in results:
    print(f"{r['size']},{r['slices']},{r['total_time']:.4f}")

# Shutdown Spark
spark.stop()
