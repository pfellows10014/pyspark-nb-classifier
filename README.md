# pyspark-nb-classifier


## Purpose

This repository contains a PySpark-based Naive Bayes classifier for text data (spam detection example). The pseudocode below describes the end-to-end pipeline (data load, preprocessing, feature extraction, training, evaluation, and saving the model).


## Assumptions

- Input is a CSV or similar text dataset with at least two columns: `label` and `message` (or `text`).
- Labels are categorical (for binary spam/ham, label values could be `spam`/`ham` or `1`/`0`).
- A Spark session is available when running the scripts (local or cluster).


## Inputs / Outputs (contract)

- Inputs: path to dataset (CSV), optional config (train/test split ratio, tokenizer options).
- Outputs: trained Naive Bayes model saved to disk, evaluation report (accuracy, precision, recall, F1, confusion matrix), optional predictions CSV.
- Error modes: missing columns, empty dataset, invalid label types.


## High-level pseudocode

1. Initialize
   - Create SparkSession
   - Set logging to WARN/ERROR to reduce noise

2. Load data
   - Read CSV/TSV with header and infer schema (or explicit schema)
   - Select/rename columns to canonical names: `label`, `text`
   - Drop or filter rows where `text` is null or empty

3. Prepare labels
   - If labels are strings, map them to numeric indices (e.g., `spam`->1, `ham`->0) using `StringIndexer` or explicit mapping

4. Text preprocessing pipeline
   - Tokenize text into words (`Tokenizer`)
   - Lowercase tokens (if tokenizer doesn't already do this)
   - Remove punctuation/non-word tokens (optional with regex tokenizer)
   - Remove stop words (`StopWordsRemover`)
   - (Optional) Apply stemming or lemmatization — note: PySpark doesn't include native stemmer; use UDF or skip for scalability

5. Split dataset
   - Train/test split (e.g., 80/20) or cross-validation folds

6. Train model
   - Create Custom `NaiveBayes` 
   - (Optional) Wrap in `Pipeline` with preprocessing and feature stages
   - Fit model on training data
   - Create a vocabulary dictionary of the distinct words in the training set
   - Calculate the prior probabilities for each class based on training labels
     - Serves as the initial guess based of the split of the training data
   - Calculate word count per class from training data
   - For each class, calculate the likelihood of each word given the class using Laplace smoothing
     - log(P(word|class)) = log((count(word, class) + alpha) / (total_word_count + alpha * Vocab size))
     - Where alpha is the smoothing parameter (commonly set to 1)

7. Evaluate model
   - Use model to transform test set and produce `predictions`
   - Calculate metrics: accuracy, precision, recall, F1 (use `MulticlassClassificationEvaluator` or compute from confusion matrix)
   - Compute confusion matrix via groupBy `label`, `prediction` counts
   - (Optional) Log per-class metrics and sample misclassified examples

8. Clean shutdown
    - Stop SparkSession


