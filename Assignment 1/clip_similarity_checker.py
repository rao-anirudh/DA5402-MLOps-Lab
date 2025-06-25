import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import io
import bson
from sklearn.metrics.pairwise import cosine_similarity


def load_clip_model():

    """
    Loads OpenAI's CLIP model for image and text embeddings
    :return model
    :return processor

    """

    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor


def load_image_from_bson(bson_data):

    """
    Loads a Python Image using binary data
    :param bson_data - Binary data
    :returns image - Python Image

    """

    image = Image.open(io.BytesIO(bson_data))
    return image


def get_embeddings_from_bson(model, processor, bson_data, caption):

    """
    Uses a CLIP model to generate embeddings for a given image and text
    :param model - CLIP model
    :param processor - CLIP processor
    :param bson_data - Binary data of image
    :param caption - Caption for the image
    :return image_embeddings
    :return text_embeddings

    """

    image = load_image_from_bson(bson_data)

    inputs = processor(text=caption, images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        image_embeddings = outputs.image_embeds
        text_embeddings = outputs.text_embeds

    return image_embeddings.tolist()[0], text_embeddings.tolist()[0]


def compute_similarity(embedding1, embedding2):

    """
    Compares two embedding vectors by cosine similarity
    :param embedding1 (list)
    :param embedding2 (list)
    :returns similarity (float)

    """

    embedding1 = np.array(embedding1).reshape(1, -1)
    embedding2 = np.array(embedding2).reshape(1, -1)
    similarity = cosine_similarity(embedding1, embedding2)[0][0]
    return similarity

