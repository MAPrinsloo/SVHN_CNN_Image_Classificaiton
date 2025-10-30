import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, spark_partition_id
from pyspark.sql.types import StringType # UDF return type
from PIL import Image
import io
import hashlib # For creating unique filenames in a distributed context
#-----------------------------------------------------------------------------------------------------------------------#
# --- Start Spark Session ---
# Ensure SparkSession is available. This will use your configured Java/PySpark setup.
spark = SparkSession.builder.appName("ImageSaverPySpark") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.cores", "2") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .getOrCreate()

# --- Define the output base directory ---
output_base_dir = "saved_images_by_class"

if not os.path.exists(output_base_dir): # This check is still for the driver (local machine)
    # --- 1. Load Parquet files with PySpark ---
    files = ['extra-00000-of-00002.parquet', 'extra-00001-of-00002.parquet']
    
    # Check if parquet files exist before trying to read them
    # Assuming files are in the current working directory or a path Spark can access
    # If they are in a different location (e.g., /mnt/e/your_data_dir/), specify full paths
    for f_path in files:
        if not os.path.exists(f_path):
            print(f"Error: Parquet file not found at {f_path}. Please ensure it's in the correct directory.")
            spark.stop()
            exit() # Exit if crucial files are missing

    df_spark_images = spark.read.parquet(*files)

    # --- 2. Extract bytes and label, handle duplicates with PySpark ---
    # `image` column is typically a structType. `image.bytes` accesses the 'bytes' field within it.
    df_spark_cleaned = df_spark_images.select(
        col("image.bytes").alias("bytes"),
        col("label").alias("label")
    ).dropDuplicates()

    # --- 3. Define UDF (User Defined Function) for saving images ---
    # This UDF will be serialized and sent to Spark executors.
    # It must contain all necessary imports inside the function if not globally available on executors.
    
    # We use `spark_partition_id()` to get a unique ID per partition to help ensure unique filenames
    # when processing in parallel. A content hash is even better for real uniqueness.
    
    def save_image_to_disk(image_data_bytes, label_val, spark_part_id):
        # Imports needed within the UDF for execution on executors
        import os
        from PIL import Image
        import io
        import hashlib

        try:
            # Ensure the output base directory is created locally on each executor's disk
            # For local mode, all executors write to the same logical filesystem
            local_output_base_dir = os.path.abspath(output_base_dir) # Get absolute path
            os.makedirs(local_output_base_dir, exist_ok=True) # Ensure base dir exists

            label_dir = os.path.join(local_output_base_dir, str(label_val))
            os.makedirs(label_dir, exist_ok=True)

            # Create a unique filename. Using a hash of the image data is robust.
            # Combine with partition ID to reduce collision risk during parallel execution.
            image_hash = hashlib.md5(image_data_bytes).hexdigest()
            filename = os.path.join(label_dir, f"image_{image_hash}_{spark_part_id}.png")

            # Convert bytes to PIL Image and save
            image = Image.open(io.BytesIO(image_data_bytes))
            image.save(filename)
            return "SUCCESS" # UDF should return a value
        except Exception as e:
            # Log errors if necessary, or return an error message
            # print(f"Error saving image: {e}") # This print goes to executor logs
            return f"FAILED: {e}"

    # Register the UDF with Spark. StringType() is the expected return type.
    save_image_udf_spark = udf(save_image_to_disk, StringType())

    # --- 4. Apply the UDF and Trigger Execution ---
    # `repartition` can help distribute the work more evenly, but for local mode it might not be strictly necessary.
    # `spark_partition_id()` is useful here for creating unique filenames across partitions.
    
    # Select the columns, apply the UDF, and add a status column.
    df_results = df_spark_cleaned.withColumn(
        "save_status",
        save_image_udf_spark(col("bytes"), col("label"), spark_partition_id())
    )

    # Trigger an action to execute the UDF and save the images.
    # .show() or .count() or .write() can trigger an action.
    # Using .foreach() or .rdd.foreachPartition() might be more direct for side-effects
    # but withColumn + write.format("noop") is a common pattern for UDF side effects.
    
    # This collects all the statuses to the driver (can be memory intensive if many statuses)
    # Alternatively, you can just print df_results.count()
    # statuses = df_results.select("save_status").collect()
    # for status_row in statuses:
    #     if "FAILED" in status_row['save_status']:
    #         print(f"Image saving failed for a row: {status_row['save_status']}")

    # The most robust way to trigger without collecting results:
    df_results.write.format("noop").mode("overwrite").save()


    print("Images have been successfully saved by class in the directory:", output_base_dir)

else:
    print(f"Download aborted - '{output_base_dir}' already exists. Delete it to re-download.")

# --- Stop Spark Session ---
spark.stop()