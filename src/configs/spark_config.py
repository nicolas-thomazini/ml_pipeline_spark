from pyspark.sql import SparkSession

# Inicializando o SparkSession
spark = SparkSession.builder \
    .appName("Solana ML Pipeline") \
    .getOrCreate()

# Verificar se o Spark foi inicializado corretamente
print(f"Spark Version: {spark.version}")