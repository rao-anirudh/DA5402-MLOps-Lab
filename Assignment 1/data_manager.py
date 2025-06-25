import pymongo
from pymongo import MongoClient
from clip_similarity_checker import get_embeddings_from_bson, compute_similarity


def connect_to_mongo_db(config):

    """
    Uses a configuration given as a Python dictionary to connect to a MongoDB cluster and
    retrieve the required collection in the database of interest
    :param config (dict) - Python dictionary containing the configuration
    :returns: collection - MongoDB collection

    """

    connection_string = config["mongo_db_connection_string"]
    cluster = MongoClient(connection_string)
    database = cluster[config["mongo_db_database"]]
    collection = database[config["mongo_db_collection"]]

    return collection


def add_data(collection, items, model, processor, image_threshold, text_threshold):

    """
    Adds data to a MongoDB collection after checking for a de-duplication constraint
    :param collection - MongoDB collection
    :param items (list) -  A list containing dictionaries of items to be added
    :param model - CLIP model
    :param processor - CLIP processor
    :param image_threshold (float) - Thumbnail similarity threshold for duplication
    :param text_threshold (float) - Headline similarity threshold for duplication

    """

    for item in items:

        current_data = collection.find({})
        current_count = collection.count_documents({})

        item["thumbnail_embedding"], item["headline_embedding"] = get_embeddings_from_bson(model, processor, item["thumbnail"], item["headline"])

        duplicate = False

        for data in current_data:

            image_similarity = compute_similarity(data["thumbnail_embedding"], item["thumbnail_embedding"])
            text_similarity = compute_similarity(data["headline_embedding"], item["headline_embedding"])

            if image_similarity > image_threshold or text_similarity > text_threshold:
                duplicate = True
                break

        if duplicate:
            pass
        else:
            item["_id"] = current_count + 1
            collection.insert_one(item)
