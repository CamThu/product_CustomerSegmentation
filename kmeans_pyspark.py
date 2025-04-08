import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, countDistinct, sum as spark_sum, max as spark_max, datediff, lit
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.clustering import KMeans
import matplotlib.pyplot as plt
import pandas as pd

spark = SparkSession.builder.appName("RFM_KMeans").getOrCreate()
df1 = pd.read_csv('Transactions.csv', parse_dates=['Date'])
df2 = pd.read_csv('Products_with_Categories.csv')
df = pd.merge(df1, df2, on='productId', how='left').fillna(0)
#  Chuyển đổi Pandas DataFrame thành PySpark DataFrame
df_spark = spark.createDataFrame(df)


df_spark = df_spark.withColumn("Date", col("Date").cast("date"))
max_date = df_spark.select(spark_max("Date")).collect()[0][0]
df_spark = df_spark.withColumn("gross_sales", col("price") * col("items"))

df_rfm = df_spark.groupBy("Member_number").agg(
    datediff(lit(max_date), spark_max("Date")).alias("Recency"),
    countDistinct("productId").alias("Frequency"),
    spark_sum("gross_sales").alias("Monetary") 
)


df_rfm = df_rfm.dropna()
# Chuẩn hóa dữ liệu với VectorAssembler
assembler = VectorAssembler(inputCols=["Recency", "Frequency", "Monetary"], outputCol="features", handleInvalid="skip")
df_rfm_vector = assembler.transform(df_rfm)

# Chuẩn hóa dữ liệu bằng StandardScaler
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
scaler_model = scaler.fit(df_rfm_vector)
df_rfm_scaled = scaler_model.transform(df_rfm_vector)



#  Chọn số cụm K bằng Elbow Method
costs = []
ks = range(2, 10)

for k in ks:
    kmeans = KMeans(featuresCol="scaled_features", k=k, seed=42)
    model = kmeans.fit(df_rfm_scaled)
    costs.append(model.summary.trainingCost)


#  Vẽ đồ thị Elbow
plt.figure(figsize=(6,4))
plt.plot(ks, costs, marker='o', linestyle='-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Cost')
plt.title('Elbow Method for Optimal K')
plt.show()


optimal_k = 5
kmeans = KMeans(featuresCol="scaled_features", k=optimal_k, seed=42)
model = kmeans.fit(df_rfm_scaled)
df_rfm_clustered = model.transform(df_rfm_scaled)


df_rfm_clustered.select("Member_number", "Recency", "Frequency", "Monetary", "prediction").show(10)


df_rfm_clustered.groupBy("prediction").agg(
    {"Recency": "mean", "Frequency": "mean", "Monetary": "mean"}
).show()



import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


df_rfm_pd = df_rfm_clustered.select("Recency", "Frequency", "Monetary", "prediction").toPandas()


# Boxplot cho từng giá trị RFM theo cụm
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.boxplot(x="prediction", y="Recency", data=df_rfm_pd, ax=axes[0])
axes[0].set_title("Boxplot - Recency")

sns.boxplot(x="prediction", y="Frequency", data=df_rfm_pd, ax=axes[1])
axes[1].set_title("Boxplot - Frequency")

sns.boxplot(x="prediction", y="Monetary", data=df_rfm_pd, ax=axes[2])
axes[2].set_title("Boxplot - Monetary")

plt.tight_layout()
plt.show()



#  3D Scatter Plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Tạo scatter plot
scatter = ax.scatter(df_rfm_pd["Recency"], df_rfm_pd["Frequency"], df_rfm_pd["Monetary"], 
                     c=df_rfm_pd["prediction"], cmap="viridis", alpha=0.6)

ax.set_xlabel("Recency")
ax.set_ylabel("Frequency")
ax.set_zlabel("Monetary")
ax.set_title("3D Scatter Plot - RFM Clusters")

# Thêm chú thích
plt.colorbar(scatter, ax=ax, label="Cluster")
plt.show()



