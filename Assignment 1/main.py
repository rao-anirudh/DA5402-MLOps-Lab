#!/usr/bin/env python3

import logging
import os
from config_reader import read_config
from google_news_scraper import scrape_google_news
from data_manager import connect_to_mongo_db, add_data
from clip_similarity_checker import load_clip_model

LOG_FILE = "script.log"
logging.basicConfig(filename=LOG_FILE,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    logging.info(f"Script started.")
    try:
        config = read_config("config.json")
        logging.info(f"Configuration file read.")
        items = scrape_google_news(config)
        logging.info(f"{len(items)} articles scraped from Google News.")
        collection = connect_to_mongo_db(config)
        current_count = collection.count_documents({})
        logging.info(f"Connected to MongoDB.")
        model, processor = load_clip_model()
        logging.info(f"Loaded CLIP model.")
        add_data(collection, items, model, processor, config["thumbnail_threshold"], config["headline_threshold"])
        new_count = collection.count_documents({})
        logging.info(f"{len(items) - (new_count - current_count)} duplicate articles detected.")
        logging.info(f"{new_count - current_count} new articles added to MongoDB.")
    except Exception as e:
        logging.error(f"An error occurred: {e}\n")
    else:
        logging.info(f"Script finished successfully.\n")


if __name__ == "__main__":
    main()
