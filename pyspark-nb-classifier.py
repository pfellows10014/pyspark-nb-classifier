import re
import math
from pyspark import RDD, Broadcast
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
from typing import List, Tuple, Any, Dict

# --- GLOBAL SPARK SESSION INITIATION ---
# Initialize Spark Session and Context
spark = SparkSession.builder.appName("SpamClassifierMapReduce").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR") # Set log level to reduce console spam

def clean_text_udf():
    """
    Returns a UDF for robust text cleaning (lowercasing, punctuation removal, 
    and ensuring minimal whitespace). Suitable for SMS/email text.
    """
    # Define the cleaning function
    def clean_text(text):
        if text is None:
            return ""
        
        text = str(text).lower()
        
        # Replace non-alphanumeric/non-space characters with a single space
        # Keep apostrophes for contractions (e.g., don't, i'm) initially
        text = re.sub(r'[^a-z0-9\s\']', ' ', text) 
        
        # More specific cleaning for potential SMS artifacts could be added here
        # e.g., removing currency symbols, handling phone numbers if needed
        
        # Collapse multiple spaces into a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing spaces
        return text.strip()
    
    # Register and return the UDF
    return udf(clean_text, StringType())

def load_and_preprocess_data(file_path: str) -> RDD:
    """
    STEP 1: Loads raw spam/ham data from CSV, selects relevant columns (v1, v2),
    cleans text, tokenizes, removes stopwords, and returns the final RDD.
    
    Input data is CSV with header. Assumes 'v1' is label, 'v2' is text.
    Output RDD: (label, [cleaned_tokens])
    """
    print(f"-> Starting Step 1: Loading and Preprocessing {file_path}")
    
    try:
        # --- LOAD CSV WITH HEADER ---
        # Adjust column names and separator as per spam.csv
        df = spark.read.csv(
            file_path,
            sep=',',            
            header=True,
            inferSchema=True,
            multiLine=True,
            escape='"'
        ).select(
            col("v1").alias("label"),      # Assuming v1 is the label (ham/spam)
            col("v2").alias("text_raw")     # Assuming v2 is the text message
        ).filter(
            # Filter where both label and text are present
            col("label").isNotNull() & col("text_raw").isNotNull()
        )
        
        # Ensure label and text columns exist
        if "label" not in df.columns or "text_raw" not in df.columns:
             raise ValueError("Required columns ('v1' as label, 'v2' as text_raw) not found in CSV header.")

        # --- TEXT PREPROCESSING PIPELINE ---
        
        # 1. Cleaning (lowercase, remove punctuation)
        cleaned_df_stage1 = df.withColumn(
            "text_clean", 
            clean_text_udf()(col("text_raw"))
        ).select("label", "text_clean") # Keep label column

        # 2. Tokenization
        tokenizer = Tokenizer(inputCol="text_clean", outputCol="raw_words")
        words_df = tokenizer.transform(cleaned_df_stage1)

        # 3. Stop Word Removal (Using default English list)
        remover = StopWordsRemover(inputCol="raw_words", outputCol="filtered_words")
        # Select final columns: label and the list of filtered words
        cleaned_df_final = remover.transform(words_df).select("label", "filtered_words") 

        # Convert final DataFrame columns (label, [list of tokens]) to RDD and Cache
        # Output RDD format: (label, [tokens])
        tokenized_rdd = cleaned_df_final.rdd.map(
            lambda row: (row.label, row.filtered_words) # Map label and filtered_words
        ) 
        tokenized_rdd.cache()
        
        print("-> Step 1 Complete: Data loaded and preprocessed.")
        return tokenized_rdd

    except FileNotFoundError:
        print(f"\n!!! FATAL ERROR: File not found at {file_path}. Ensure '{file_path}' is in the correct directory.")
        return None # Return None to indicate failure
    except Exception as e:
        print(f"\n!!! FATAL ERROR during Step 1 execution.")
        print(f"Error details: {e}")
        return None # Return None on other errors
    
def calculate_log_likelihoods(
        log_priors: Dict[str, float],
        word_counts_map: Dict[Tuple[str, str], int],
        total_words_per_class: Dict[str, int],
        vocabulary_size: int,
        LAPLACE_SMOOTHING: float
) -> Dict[Tuple[str, str], float]:
    """
    Calculates the log-likelihood P(Word | Class) for each word in the vocabulary,
    applying Laplace smoothing. Includes handling for unseen words.
    """
    print("\nCalculating log likelihoods...")
    log_likelihoods = {} # Format: {(label, word): log_probability}

    all_words_in_train = set(word for (label, word) in word_counts_map.keys())
    print(f"Total unique words found in word_counts_map: {len(all_words_in_train)}")

    for label in log_priors.keys():
        total_words_in_class = total_words_per_class.get(label, 0)
        
        # Calculate log likelihood for each word in the vocabulary
        for word in all_words_in_train:
            count_w_c = word_counts_map.get((label, word), 0)
            # Apply Laplace smoothing
            prob_w_c = (count_w_c + LAPLACE_SMOOTHING) / (total_words_in_class + LAPLACE_SMOOTHING * vocabulary_size)
            log_likelihoods[(label, word)] = math.log(prob_w_c)
        
        # Handle unseen words for this class
        prob_unseen_w_c = LAPLACE_SMOOTHING / (total_words_in_class + LAPLACE_SMOOTHING * vocabulary_size)
        log_likelihoods[(label, "__UNSEEN__")] = math.log(prob_unseen_w_c)

    return log_likelihoods

def classify_document(
        record: Tuple[str, List[str]], 
        model_bcast: Broadcast[Dict[str, Any]]
    ) -> Tuple[str, str]:
    """
    Classifies a single document using the Naive Bayes model parameters.
    
    Input:
        record: (actual_label, [tokens])
        model_bcast: Broadcasted model parameters containing log_priors and log_likelihoods.
    
    Output:
        (actual_label, predicted_label)
    """
    actual_label, tokens = record
    model_params = model_bcast.value # Extract the dictionary from the broadcast variable
    
    log_priors = model_params["log_priors"]
    log_likelihoods = model_params["log_likelihoods"]
    classes = model_params["classes"] # Get list of class names ('spam', 'ham')
    
    scores = {} # Dictionary to hold the calculated score for each class for this document
    
    # Calculate the Naive Bayes score for each possible class
    for label in classes:
        # Starting with the log prior probability of the class
        class_score = log_priors.get(label, -float('inf')) 
        
        # Iterate through each token in the current document's token list
        for token in tokens:
            token_log_likelihood = log_likelihoods.get(
                (label, token), 
                log_likelihoods.get((label, "__UNSEEN__"), -float('inf')) 
            )
            
            # Add the token's log likelihood to the class's score
            class_score += token_log_likelihood
            
        # Store the final calculated score for this class
        scores[label] = class_score
        
    # Determine the predicted class by finding the class label with the maximum score
    if not scores: 
        # Handle edge case: if no scores were calculated (e.g., empty token list, missing classes)
        predicted_label = "unknown" # Assign a default or handle as error
    else:
        # The predicted label is the key (class name) associated with the highest value (score)
        predicted_label = max(scores, key=scores.get)
        
    # Return the actual label and the predicted label for this document
    return (actual_label, predicted_label)

if __name__ == "__main__":
    
    # --- CONFIGURATION ---
    # File name for the spam dataset (ensure it's in the same directory or provide full path)
    DATA_FILE = "spam.csv" 
    
    print("--- Starting Naive Bayes Spam Classifier Pipeline ---")
    
    tokenized_rdd = None
    N_docs = 0
    train_rdd = None
    test_rdd = None
    train_count = 0
    test_count = 0
    model_parameters = None # Will store the trained model

    try:
        # --- STEP 1: DATA PREPROCESSING ---
        tokenized_rdd = load_and_preprocess_data(DATA_FILE)
        
        if tokenized_rdd is not None:
             N_docs = tokenized_rdd.count()
             if N_docs == 0:
                 print("\n!!! ERROR: Preprocessing resulted in an empty RDD. Check data file and preprocessing logic.")
                 raise ValueError("Preprocessing failed.") # Stop execution if RDD is empty
        else:
             # Error handled within load_and_preprocess_data
             raise ValueError("Data loading/preprocessing failed.") # Stop execution if loading fails

        print(f"✅ Preprocessing Success! Total documents loaded: {N_docs}")
            
        # Print a sample of the preprocessed RDD
        print("\nSample of Preprocessed RDD (Label, [Tokens]):") 
        for item in tokenized_rdd.take(3):
            label = item[0] 
            tokens = item[1] 
            print(f"  Label: {label}, Tokens: {tokens[:10]}...") 

        # --- STEP 2: SPLITTING THE DATA ---
        print("\n--- Starting Step 2: Splitting Data (80% Train, 20% Test) ---")
        train_rdd, test_rdd = tokenized_rdd.randomSplit([0.8, 0.2], seed=42)
        train_rdd.cache()
        test_rdd.cache()
            
        train_count = train_rdd.count()
        test_count = test_rdd.count()
        print(f"Training set size: {train_count} documents")
        print(f"Testing set size: {test_count} documents")
        print("✅ Step 2 Success! Data split into training and testing sets.")
        print("---------------------------------------------------------------")

        # Check if train or test set is empty
        if train_count == 0 or test_count == 0:
            raise ValueError("Training or testing set is empty after split. Check data or split ratio.")

        # --- STEP 3: TRAINING THE MODEL (MapReduce Implementation) ---
        print("\n--- Starting Step 3: Training the Naive Bayes Model ---")
        
        # 1. Calculate Vocabulary Size (M) from train_rdd
        print("Calculating vocabulary size (M)...")
        # Flatten the list of tokens from each document into a single RDD of words
        # Get distinct words and count them
        vocabulary_rdd = train_rdd.flatMap(lambda x: x[1]).distinct()
        vocabulary_size = vocabulary_rdd.count()
        # No need to collect the full vocabulary set here unless needed later
        # vocabulary = set(vocabulary_rdd.collect()) 
        print(f"Vocabulary Size (M): {vocabulary_size}")

        # Define Laplace smoothing constant (usually 1)
        LAPLACE_SMOOTHING = 1.0

        # --- Placeholder for remaining Step 3 parts ---
        print("Calculating class priors...")
        # 2. Calculate Class Priors (log P(Ck))
        #    - map: (label, [tokens]) -> (label, 1)
        #    - reduceByKey: Sum counts per label
        #    - collectAsMap: Get {label: count} dictionary
        #    - Calculate log_priors = {label: log(count / train_count)}
        class_counts = train_rdd.map(lambda x: (x[0], 1)).reduceByKey(lambda a, b: a + b).collectAsMap()

        # Calculate log priors for each class
        # log P(Ck) = log(Count(Ck) / TotalDocs)
        log_priors = {
            label: math.log(count / float(train_count)) 
            for label, count in class_counts.items()
        }

        print("Class Counts:", class_counts)
        print("Log Priors:", {k: f"{v:.4f}" for k, v in log_priors.items()}) 

        print("Calculating word counts per class...")
        # 3. Calculate Word Counts per Class (Count(Wi, Ck))
        #    - flatMap: (label, [w1, w2]) -> [ ((label, w1), 1), ((label, w2), 1) ]
        #    - reduceByKey: Sum counts for each (label, word)
        #    - collectAsMap: Get {(label, word): count} dictionary
        word_counts_map = train_rdd.flatMap(
            lambda x: [((x[0], word), 1) for word in x[1]]
        ).reduceByKey(lambda a, b: a + b).collectAsMap()

        print("Calculating total words per class...")
        # 4. Calculate Total Words per Class (TotalWords(Ck))
        #    - map: (label, [tokens]) -> (label, len(tokens))
        #    - reduceByKey: Sum len(tokens) per label
        #    - collectAsMap: Get {label: total_words} dictionary
        total_words_per_class = train_rdd.map(
            lambda x: (x[0], len(x[1]))
        ).reduceByKey(lambda a, b: a + b).collectAsMap()

        print("Calculating log likelihoods...")
        # 5. Calculate Log Likelihoods with Laplace Smoothing
        log_likelihoods = calculate_log_likelihoods(
            log_priors, 
            word_counts_map, 
            total_words_per_class, 
            vocabulary_size, 
            LAPLACE_SMOOTHING
        )
        
        model_parameters = {
            "log_priors": log_priors,
            "log_likelihoods": log_likelihoods,
            "vocabulary_size": vocabulary_size,
            "classes": list(log_priors.keys()) # Get classes from priors dict
        }

        # --- STEP 4: TESTING MODEL AND EVALUATION ---
        print("\n--- Placeholder for Step 4: Testing Model and Evaluation ---")

        print("Broadcasting model parameters to workers...")
        model_bcast = sc.broadcast(model_parameters)
        
        print("Applying model to test data...")
        predictions_rdd = test_rdd.map(
            lambda record: classify_document(record, model_bcast)
        )
        
        print("Collecting results...")
        results = predictions_rdd.collect()
        
        print("Calculating accuracy...")
        correct_predictions = sum(1 for actual, predicted in results if actual == predicted)
        total_predictions = test_count

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        print("\n" + "="*50)
        print("       Naive Bayes MapReduce - Final Results")
        print("="*50)
        print(f"Total Test Documents: {total_predictions}")
        print(f"Correctly Classified: {correct_predictions}")
        print(f"Model Accuracy: {accuracy:.4f}")
        print("-" * 50)
        print("Class Priors (Log):")
        for label, log_prior in log_priors.items():
            prior_prob = math.exp(log_prior) # Convert log prior back to probability
            print(f"  Class '{label}': {log_prior:.4f} (Prior: {prior_prob:.4f})")
        print("="*50)
        print("✅ Step 4 Complete. Evaluation finished.")
        print("---------------------------------------------------------------")
        
        # --- CLEANUP ---
        tokenized_rdd.unpersist()
        train_rdd.unpersist() 
        test_rdd.unpersist() # Unpersist test RDD after evaluation

    except Exception as e:
        print(f"\n!!! FATAL ERROR during pipeline execution.")
        print(f"Error details: {e}")
        # Attempt to unpersist RDDs even on error
        if tokenized_rdd: tokenized_rdd.unpersist()
        if train_rdd: train_rdd.unpersist()
        if test_rdd: test_rdd.unpersist()
        
    finally:
        # Stop Spark Session
        print("\nStopping Spark session...")
        spark.stop()
        print("Spark session stopped.")