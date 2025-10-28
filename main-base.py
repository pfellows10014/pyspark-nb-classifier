import re
import math
from pyspark import RDD
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.sql.functions import col, udf, concat_ws
from pyspark.sql.types import StringType
from typing import List, Tuple, Any

# --- GLOBAL SPARK SESSION INITIATION ---
# Initialize Spark Session and Context
spark = SparkSession.builder.appName("ReusableTextPreprocessing").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR") # Set log level to reduce console spam

def clean_text_udf():
    """
    Returns a UDF for robust text cleaning (lowercasing, punctuation removal, 
    and ensuring minimal whitespace).
    """
    # Define the cleaning function
    def clean_text(text):
        if text is None:
            return ""
        
        text = str(text).lower()
        
        # Replace non-alphanumeric/non-space characters with a single space
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # Collapse multiple spaces into a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing spaces
        return text.strip()
    
    # Register and return the UDF
    return udf(clean_text, StringType())

def load_and_preprocess_data(file_path: str) -> RDD:
    """
    Loads raw data, performs cleaning, tokenizes, removes stopwords, 
    and returns the final tokenized RDD.
    
    Input data is assumed to be TSV/CSV format: (ID/Label, Text).
    Output RDD: (document_id, [cleaned_tokens])
    """
    print(f"-> Loading text data from {file_path}")
    
    # --- STEP 0: LOAD TSV/CSV WITH NO HEADER ---
    # Assumes data is tab-separated with label/ID in _c0 and text in _c1
    df = spark.read.csv(
        file_path,
        sep='\t',            
        header=False         
    ).select(
        col("_c0").alias("doc_id"),        
        col("_c1").alias("text_raw") 
    ).filter(col("text_raw").isNotNull()) 

    # --- 1. CLEANING ---
    # Apply Punctuation Cleaning UDF
    plots_cleaned_df = df.withColumn(
        "text_clean", 
        clean_text_udf()(col("text_raw"))
    ).select("doc_id", "text_clean")

    # --- 2. TOKENIZATION ---
    tokenizer = Tokenizer(inputCol="text_clean", outputCol="raw_words")
    words_df = tokenizer.transform(plots_cleaned_df)

    # --- 3. STOP WORD REMOVAL (Using default list, add customs here if needed) ---
    remover = StopWordsRemover(inputCol="raw_words", outputCol="filtered_words")
    cleaned_df = remover.transform(words_df).select("doc_id", "filtered_words")

    # Convert final DataFrame columns (doc_id, [list of tokens]) to RDD and Cache
    # Output RDD format: (name, [tokens])
    tokenized_rdd = cleaned_df.rdd.map(tuple)
    tokenized_rdd.cache()
    
    return tokenized_rdd

if __name__ == "__main__":
    
    # --- CONFIGURATION FOR TEST RUN ---
    # NOTE: Replace 'path/to/your/data.txt' with the actual path and filename of your dataset.
    DATA_FILE = "path/to/your/data.txt" 
    
    print("--- Starting Text Preprocessing Pipeline ---")
    
    try:
        # Load and preprocess text data
        tokenized_rdd = load_and_preprocess_data(DATA_FILE)
        N_docs = tokenized_rdd.count()
        
        if N_docs > 0:
            print(f"✅ Preprocessing Success! Total documents loaded: {N_docs}")
            
            # Print a sample of the preprocessed RDD for verification
            print("\nSample of Preprocessed RDD (Doc ID/Label, [Tokens]):")
            for item in tokenized_rdd.take(3):
                doc_id = item[0]
                tokens = item[1]
                print(f"  ID: {doc_id}, Tokens: {tokens[:10]}...") # Show first 10 tokens
            
            # Clean up cache
            tokenized_rdd.unpersist()

        else:
            print("\n!!! ERROR: RDD is empty. Check your data file path and format.")
            
    except Exception as e:
        print(f"\n!!! FATAL ERROR during execution.")
        print(f"Error details: {e}")
        
    finally:
        # Stop Spark Session
        spark.stop()
