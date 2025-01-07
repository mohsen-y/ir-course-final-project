from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from core import content_manager, ir_manager, nltk_resource_manager
from router import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Downloading NLTK resources...")
    nltk_resource_manager.download_resources()

    logger.info("Loading documents...")
    content_manager.load_documents()

    logger.info("Loading ground truth...")
    content_manager.load_ground_truth()

    logger.info("Loading boolean queries...")
    content_manager.load_boolean_queries()

    logger.info("Loading queries...")
    content_manager.load_queries()

    logger.info("Training Word2Vec model...")
    ir_manager.train_w2v_model(list(content_manager.documents.values()))

    logger.info("Creating documents' vectors...")
    ir_manager.create_documents_vectors(content_manager.documents)

    logger.info("Creating inverted index...")
    ir_manager.create_inverted_index(content_manager.documents)

    yield


app = FastAPI(debug=True, lifespan=lifespan)

app.include_router(router, prefix="/api")
