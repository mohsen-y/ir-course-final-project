import re
from enum import Enum

from pydantic import BaseModel


class ContentSectionEnum(str, Enum):
    id = "id"
    title = "title"
    author = "author"
    body = "body"
    reference = "reference"
    cross_references = "cross_references"


class ContentSectionRegExEnum(Enum):
    id = re.compile(r"^\.I\s(\d+)$")
    title = re.compile(r"^\.T$")
    author = re.compile(r"^\.A$")
    body = re.compile(r"^\.W$")
    reference = re.compile(r"^\.B$")
    cross_references = re.compile(r"^\.X$")


class SetOperationEnum(str, Enum):
    not_op = "not_op"
    and_op = "and_op"
    or_op = "or_op"


class Document(BaseModel):
    title: str
    author: str
    abstract: str


class QueryResult(BaseModel):
    id: int
    documents: list[int]


class W2VModelEvaluation(BaseModel):
    mean_average_precision: float
