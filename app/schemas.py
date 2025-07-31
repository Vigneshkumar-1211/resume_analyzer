# app/schemas.py
from typing import List, Dict, Optional, Union

from pydantic import BaseModel, Field

class DetailedScore(BaseModel):
    criterion: str = Field(description="The name of the sub-criterion being scored.")
    score: int = Field(description="The score for this specific sub-criterion.")
    
    score_justification: Optional[str] = Field(None, description="Justification for why this score was given.")
    suggestion: Optional[str] = Field(None, description="Actionable suggestion for how to improve this area.")
    wrong_thing: Optional[Union[str, List[str]]] = Field(None, description="The specific issue or text from the resume that caused a score reduction.")
class FinalReport(BaseModel):
    """Data model for the final, comprehensive JSON report."""
    total_score: str = Field(description="The total aggregated score out of 100, formatted as 'score/100'.")
    category_scores: Dict[str, str] = Field(description="A dictionary mapping each category to its total score, formatted as 'score/max_score'.")
    detailed_breakdown: Dict[str, List[DetailedScore]] = Field(description="A nested dictionary containing the detailed scoring for each category.")
    parsed_resume_content: Dict = Field(description="The original structured JSON parsed from the resume PDF.")