
from pydantic import BaseModel
from typing import Any, List, Optional

class RulesRequest(BaseModel):
    data: List
    predictions: List
    epsilon: float

class Rule(BaseModel):
    coef: List
    intercept: List
    threshold: float
    quad_coef: Optional[List] = None

class RulesResponse(BaseModel):
    rules: List[Rule]
    rule_values: List[Any]
    rule_summary: List