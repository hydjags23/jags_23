import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, Normalizer
from pyspark.sql.functions import concat, col, lit, lower, udf
from pyspark.sql.types import DoubleType
import numpy as np

# Initialize Spark session
spark = SparkSession.builder.appName("TF-IDF Recommendation System").getOrCreate()

# Define the cosine similarity function
def cosine_similarity(vector1, vector2):
    return float(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))

# Streamlit UI
st.title("Restaurant Recommendation System")

# User input for city and establishment type
user_city = st.text_input("Enter the city", "chennai").lower()
user_establishment = st.text_input("Enter the establishment type", "fine dining").lower()

# Load data
@st.cache(allow_output_mutation=True)
def load_data():
    df = spark.read.csv("/content/drive/My Drive/zomato/zomato.csv", header=True, inferSchema=True)
    df = df.fillna({'cuisines': '', 'filtered_highlights': '', 'establishment': ''})
    df = df.withColumn('enhanced_text', concat(col('cuisines'), lit(" "), col('filtered_highlights'), lit(" "), col('establishment'), lit(" "),))
    return df

df = load_data()

# Process data for TF-IDF
tokenizer = Tokenizer(inputCol="enhanced_text", outputCol="words")
wordsData = tokenizer.transform(df)

hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=1024)
featurizedData = hashingTF.transform(wordsData)

idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

normalizer = Normalizer(inputCol="features", outputCol="normFeatures")
normData = normalizer.transform(rescaledData)

# Filtering data based on user input
def get_recommendations(city, establishment):
    filtered_df = normData.filter(
        (col("city") == city) & 
        (col("establishment") == establishment)
    )
    
    if filtered_df.rdd.isEmpty():
        return "No entries found for specified city and establishment type."
    
    example_feature_vector = spark.sparkContext.broadcast(filtered_df.first()['normFeatures'])

    # Define a UDF for cosine similarity using the broadcast variable
    cosine_similarity_udf_bc = udf(lambda vector: cosine_similarity(vector, example_feature_vector.value), DoubleType())
    filtered_df = filtered_df.withColumn("similarity", cosine_similarity_udf_bc(col("normFeatures")))
    recommended_df = filtered_df.select("name", "aggregate_rating", "average_cost_for_two").orderBy("similarity", ascending=False).limit(10)
    
    return recommended_df

# Display button to get recommendations
if st.button('Get Recommendations'):
    recommendations = get_recommendations(user_city, user_establishment)
    if isinstance(recommendations, str):
        st.write(recommendations)
    else:
        st.write(recommendations.toPandas())

# Close Spark session on script end
st.on_session_end(spark.stop)
