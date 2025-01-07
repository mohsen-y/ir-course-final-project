import ast
import json
import numpy as np
import pathlib
import sys

import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from scipy import spatial

import schemas
import utils
from schemas import ContentSectionEnum, ContentSectionRegExEnum


class NLTKResourceManager:
    def __init__(self) -> None:
        self._data_dir = pathlib.Path(sys.prefix) / "nltk_data"

    def download_resources(self) -> None:
        nltk.download("stopwords", download_dir=self._data_dir)
        nltk.download("punkt_tab", download_dir=self._data_dir)
        nltk.download("wordnet", download_dir=self._data_dir)


class IRManager:
    def __init__(self) -> None:
        self._lemmatizer = nltk.stem.WordNetLemmatizer()
        self._w2v_model: Word2Vec = None
        self._w2v_model_config = {
            "vector_size": 100,
            "window": 5,
            "min_count": 1,
            "workers": 1,
        }
        self._english_stopwords = set(stopwords.words("english"))
        self._documents_vectors: dict[int, np.ndarray] = dict()
        self._inverted_index: dict[str, list[int]] = dict()
        self._documents_ids: set[int] = set()

    @property
    def w2v_model(self) -> Word2Vec:
        return self._w2v_model

    @property
    def inverted_index(self) -> dict[str, list[int]]:
        return self._inverted_index

    @property
    def documents_vectors(self) -> dict[int, np.ndarray]:
        return self._documents_vectors

    def train_w2v_model(self, documents: list[schemas.Document]) -> None:
        documents_sentences: list[list[str]] = []

        for document in documents:
            content = (document.title + document.author + document.abstract).lower()
            sentences = nltk.tokenize.sent_tokenize(content)
            for sentence in sentences:
                tokens = [
                    self._lemmatizer.lemmatize(token)
                    for token in nltk.tokenize.word_tokenize(sentence)
                    if token not in self._english_stopwords
                ]
                documents_sentences.append(tokens)

        self._w2v_model = Word2Vec(documents_sentences, **self._w2v_model_config)

    def create_documents_vectors(self, documents: dict[int, schemas.Document]) -> None:
        for doc_id, document in documents.items():
            content = (document.title + document.author + document.abstract).lower()
            words = [
                self._lemmatizer.lemmatize(token)
                for token in nltk.tokenize.word_tokenize(content)
                if token not in self._english_stopwords and token in self._w2v_model.wv
            ]
            self._documents_vectors[doc_id] = sum(
                self._w2v_model.wv[word] for word in words
            ) / len(words)

    def create_inverted_index(self, documents: dict[int, schemas.Document]) -> None:
        for doc_id, document in documents.items():
            self._documents_ids.add(doc_id)
            for attr in ["title", "author", "abstract"]:
                for token in nltk.tokenize.word_tokenize(
                    getattr(document, attr).lower()
                ):
                    if token not in self._english_stopwords:
                        token = self._lemmatizer.lemmatize(token)
                        postings = self.inverted_index.setdefault(token, [])
                        postings.append(doc_id)

    def _create_vector(self, content: str) -> np.ndarray:
        words = [
            self._lemmatizer.lemmatize(token)
            for token in nltk.tokenize.word_tokenize(content.lower())
            if token not in self._english_stopwords and token in self._w2v_model.wv
        ]
        return sum(self._w2v_model.wv[word] for word in words) / len(words)

    def retrieve_docs_from_inverted_index(self, query: ast.expr) -> list[int]:
        return utils.evaluate_boolean_query(
            query,
            inverted_index=self._inverted_index,
            lemmatizer=self._lemmatizer,
            all_docs=self._documents_ids,
        )

    def retrieve_docs_by_w2v_model(self, query: str) -> list[int]:
        query_vec = self._create_vector(query)

        similarities = []

        for doc_id, doc_vec in self._documents_vectors.items():
            similarity = spatial.distance.cosine(query_vec, doc_vec)
            similarities.append((doc_id, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)

        return [doc_id for doc_id, _ in similarities]

    def _calculate_average_precision(
        self, retrieved_docs: list[int], relevant_docs: list[int]
    ) -> float:
        relevant_count = 0
        precision_sum = 0.0

        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_docs:
                relevant_count += 1
                precision = relevant_count / (i + 1)
                precision_sum += precision
        
        average_precision = precision_sum / len(relevant_docs) if relevant_docs else 0

        return average_precision

    def calculate_mean_ap(
        self, query_results: dict[int, list[int]], ground_truth: dict[int, list[int]]
    ) -> float:
        average_precisions = []

        for query_id, retrieved_docs in query_results.items():
            relevant_docs = ground_truth.get(query_id, [])
            ap = self._calculate_average_precision(retrieved_docs, relevant_docs)
            average_precisions.append(ap)
        
        mean_average_precision = sum(average_precisions) / len(average_precisions)

        return mean_average_precision


class ContentManager:
    def __init__(self) -> None:
        self._documents: dict[int, schemas.Document] = dict()
        self._ground_truth: dict[int, list[int]] = dict()
        self._boolean_queries: dict[int, ast.expr] = dict()
        self._queries: dict[int, str] = dict()

    @property
    def documents(self) -> dict[int, schemas.Document]:
        return self._documents

    @property
    def ground_truth(self) -> dict[int, list[int]]:
        return self._ground_truth

    @property
    def boolean_queries(self) -> dict[int, str]:
        return self._boolean_queries

    @property
    def queries(self) -> dict[int, str]:
        return self._queries

    def load_documents(self) -> None:
        with open("./data/CISI.ALL", mode="r") as file:
            current_doc_id: int = None
            current_doc_section: ContentSectionEnum = None

            for line in file:
                line = line.strip()

                if match := ContentSectionRegExEnum.id.value.match(line):
                    current_doc_id = int(match.group(1))
                    self._documents[current_doc_id] = schemas.Document(
                        title="", author="", abstract=""
                    )

                elif ContentSectionRegExEnum.title.value.match(line):
                    current_doc_section = ContentSectionEnum.title

                elif ContentSectionRegExEnum.author.value.match(line):
                    current_doc_section = ContentSectionEnum.author

                elif ContentSectionRegExEnum.body.value.match(line):
                    current_doc_section = ContentSectionEnum.body

                elif ContentSectionRegExEnum.cross_references.value.match(line):
                    current_doc_section = ContentSectionEnum.cross_references

                else:
                    line = line.replace("\n", " ")

                    match current_doc_section:
                        case ContentSectionEnum.title:
                            self._documents[current_doc_id].title += line

                        case ContentSectionEnum.author:
                            self._documents[current_doc_id].author += line

                        case ContentSectionEnum.body:
                            self._documents[current_doc_id].abstract += line

    def load_ground_truth(self) -> None:
        with open("./data/CISI.REL", mode="r") as file:
            for line in file:
                query_id, doc_id = map(int, line.split()[:2])
                documents = self._ground_truth.setdefault(query_id, [])
                documents.append(doc_id)

    def load_boolean_queries(self) -> None:
        with open("./data/CISI.BLN.JSON", mode="r") as file:
            queries = json.load(file)

        for query_id, query_str in queries.items():
            query = (
                query_str.replace("#and", "and_op")
                .replace("#or", "or_op")
                .replace("#not", "not_op")
            )
            self._boolean_queries[int(query_id)] = ast.parse(query, mode="eval").body

    def load_queries(self) -> None:
        with open("./data/CISI.QRY", mode="r") as file:
            current_query_id: int = None
            current_query_section: ContentSectionEnum = None

            for line in file:
                line = line.strip()

                if match := ContentSectionRegExEnum.id.value.match(line):
                    current_query_id = int(match.group(1))
                    self._queries[current_query_id] = ""

                elif ContentSectionRegExEnum.title.value.match(line):
                    current_query_section = ContentSectionEnum.title

                elif ContentSectionRegExEnum.author.value.match(line):
                    current_query_section = ContentSectionEnum.author

                elif ContentSectionRegExEnum.body.value.match(line):
                    current_query_section = ContentSectionEnum.body

                elif ContentSectionRegExEnum.reference.value.match(line):
                    current_query_section = ContentSectionEnum.reference

                elif current_query_section is ContentSectionEnum.body:
                    line = line.replace("\n", " ")
                    self._queries[current_query_id] += line


nltk_resource_manager = NLTKResourceManager()
content_manager = ContentManager()
ir_manager = IRManager()
