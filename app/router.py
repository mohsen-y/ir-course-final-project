from fastapi import APIRouter

import schemas
from core import content_manager, ir_manager

router = APIRouter(prefix="/documents")


@router.get("/boolean", response_model=list[schemas.QueryResult])
def retrieve_docs_from_inverted_index():
    results = []
    for query_id, query in content_manager.boolean_queries.items():
        documents = ir_manager.retrieve_docs_from_inverted_index(query)
        results.append(schemas.QueryResult(id=query_id, documents=documents))
    return results


@router.get("/w2v", response_model=list[schemas.QueryResult])
def retrieve_docs_by_w2v_model():
    results = []
    for query_id, query in content_manager.queries.items():
        documents = ir_manager.retrieve_docs_by_w2v_model(query)
        results.append(schemas.QueryResult(id=query_id, documents=documents[:10]))
    return results


@router.post("/w2v/evaluation", response_model=schemas.W2VModelEvaluation)
def evaluate_w2v_model(query_results: list[schemas.QueryResult]):
    results = {result.id: result.documents for result in query_results}
    mean_ap = ir_manager.calculate_mean_ap(
        results, content_manager.ground_truth
    )
    return schemas.W2VModelEvaluation(mean_average_precision=mean_ap)
