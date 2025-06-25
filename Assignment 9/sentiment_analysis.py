import logging
from pyspark import SparkContext, SparkConf
import random


# Set up logging configuration to capture script activities in a log file
LOG_FILE = "script.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_review_block(block):
    """
    Parse a single review block from the input file, extracting the review score and text.

    Args:
        block (str): A block of text containing review data, typically in a key-value format.

    Returns:
        tuple: A tuple containing the review score (float) and the review text (str).
               If parsing fails, returns None.
    """
    lines = block.strip().split("\n")
    data = {}
    for line in lines:
        if ":" in line:
            key, val = line.split(":", 1)
            data[key.strip()] = val.strip()

    # Attempt to extract the score and text; if an error occurs, log it and return None
    try:
        score = float(data['review/score'])
        text = data['review/text']
        return score, text
    except Exception as e:
        logging.error(f"Error while parsing review: {block[:50]}... | {e}")
        return None


def apply_sentiment_partition(iterator):
    """
    Apply sentiment analysis to a partition of reviews using a transformer model.

    Args:
        iterator (iterable): An iterator over a batch of reviews (score, text).

    Yields:
        tuple: A tuple containing the review score and predicted sentiment label ("POSITIVE" or "NEGATIVE").
               If sentiment analysis fails, yield the review with "ERROR" as the sentiment.
    """
    from transformers import pipeline
    sentiment_pipeline = pipeline("sentiment-analysis", truncation=True)

    # Iterate through each review in the partition, performing sentiment analysis
    for score, text in iterator:
        try:
            sentiment = sentiment_pipeline(text)[0]['label']
        except Exception as e:
            logging.error(f"Error during sentiment analysis: {text[:50]}... | {e}")
            sentiment = "ERROR"
        yield score, sentiment


def partition_confusion_matrix(iterator):
    """
    Calculate the confusion matrix components (TP, FP, FN, TN) for a partition of sentiment results.

    Args:
        iterator (iterable): An iterator over the sentiment analysis results (score, predicted sentiment).

    Yields:
        tuple: A tuple containing the counts of True Positives (TP), False Positives (FP),
               False Negatives (FN), and True Negatives (TN) in the partition.
    """
    tp = fp = fn = tn = 0

    # Process each review and its prediction to count TP, FP, FN, TN
    for score, pred in iterator:
        if pred == "ERROR":
            continue  # Skip reviews where sentiment analysis failed

        true_label = "POSITIVE" if score >= 3.0 else "NEGATIVE"
        if true_label == "POSITIVE" and pred == "POSITIVE":
            tp += 1
        elif true_label == "POSITIVE" and pred == "NEGATIVE":
            fn += 1
        elif true_label == "NEGATIVE" and pred == "POSITIVE":
            fp += 1
        elif true_label == "NEGATIVE" and pred == "NEGATIVE":
            tn += 1
    yield tp, fp, fn, tn


def run_spark_pipeline(num_reviews):
    """
    Set up the Spark pipeline for processing reviews, applying sentiment analysis,
    and calculating evaluation metrics like precision and recall.

    Args:
        num_reviews (int): The number of reviews to process from the input file.
    """
    # Set up Spark context and configuration
    conf = SparkConf().setAppName("GourmetReviewSentimentRDD")
    sc = SparkContext(conf=conf)

    # Set seed for reproducibility
    random.seed(5402)
    logging.info("Random seed set to 5402")

    # Parse and randomly sample reviews
    logging.info("Reading and parsing reviews...")
    with open("Gourmet_Foods.txt", "r", encoding="utf-8") as f:
        content = f.read()
    blocks = content.strip().split("\n\n")
    parsed_reviews = random.sample(list(filter(None, map(parse_review_block, blocks))), k=num_reviews)
    logging.info(f"{len(parsed_reviews)} reviews parsed")
    rdd = sc.parallelize(parsed_reviews)

    # Perform sentiment analysis on the reviews using mapPartitions
    logging.info("Running sentiment prediction...")
    results_rdd = rdd.mapPartitions(apply_sentiment_partition)
    logging.info("Sentiment prediction complete")

    # Evaluate the predictions by calculating the confusion matrix (TP, FP, FN, TN)
    logging.info("Evaluating predictions...")
    conf_matrix_rdd = results_rdd.mapPartitions(partition_confusion_matrix)

    # Reduce the confusion matrix counts across all partitions
    final_counts = conf_matrix_rdd.reduce(lambda x, y: tuple(a + b for a, b in zip(x, y)))
    TP, FP, FN, TN = final_counts

    # Calculate precision and recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Print confusion matrix and metrics
    print(f"Confusion Matrix:")
    print(f"               Predicted")
    print(f"               POS    NEG")
    print(f"Actual POS     {TP:<7}{FN}")
    print(f"Actual NEG     {FP:<7}{TN}")
    print()
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")

    logging.info(f"Evaluation complete | Confusion Matrix: TP={TP}, FP={FP}, FN={FN}, TN={TN} | Precision={precision:.4f} | Recall={recall:.4f}")


if __name__ == "__main__":
    # Run the Spark pipeline for a subset of reviews (e.g., 1000 reviews)
    run_spark_pipeline(num_reviews=1000)
