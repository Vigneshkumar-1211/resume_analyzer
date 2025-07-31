# app/core/analyzer.py
import os
import json
import logging
from typing import Dict, List

import google.generativeai as genai
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from app.schemas import DetailedScore

# Logger setup
logger = logging.getLogger(__name__)

def clean_json_string(json_str: str) -> str:
    start_bracket = json_str.find('[')
    start_brace = json_str.find('{')
    if start_bracket == -1 and start_brace == -1: return json_str
    start_index = min(start_bracket, start_brace) if start_bracket != -1 and start_brace != -1 else max(start_bracket, start_brace)
    end_bracket = json_str.rfind(']')
    end_brace = json_str.rfind('}')
    end_index = max(end_bracket, end_brace)
    if start_index != -1 and end_index != -1 and end_index > start_index:
        return json_str[start_index:end_index+1]
    return json_str

def parse_llm_json_list(text: str) -> List[Dict]:
    """Cleans and parses a JSON list string from LLM output."""
    try:
        cleaned_str = clean_json_string(text)
        return json.loads(cleaned_str)
    except json.JSONDecodeError:
        logger.warning(f"Could not decode JSON list from text: {text}")
        return []

def parse_resume_from_pdf(pdf_bytes: bytes, filename: str) -> Dict:
    """Takes PDF bytes, uses Gemini to parse it, and returns a dictionary."""
    logger.info("--- Running Step 1: Parsing PDF with Gemini ---")
    try:
        pdf_file_part = {'mime_type': 'application/pdf', 'data': pdf_bytes}
        model = genai.GenerativeModel(model_name='gemini-1.5-flash')

        parsing_prompt = """
You are a specialized AI assistant for HR and recruitment. Your single purpose is to parse a resume with extreme precision and convert it into a structured JSON object.

**Instructions:**
1.  Analyze the entire resume text provided.
2.  Extract the information and map it to the JSON schema below.
3.  **Your output must be ONLY the valid JSON object.** Do not include any introductory text, explanations, or markdown formatting like ```json. Your response must start with `{` and end with `}`.
4.  If a specific piece of information or an entire section (e.g., 'projects') is not found in the resume, you MUST use `null` as the value for that field. Do not omit the key.

**JSON Schema:**
{
  "fullName": "string",
  "contactInformation": {
    "email": "string",
    "phone": "string",
    "linkedin": "string",
    "website": "string"
  },
  "professionalSummary": "string",
  "workExperience": [
    {
      "jobTitle": "string",
      "company": "string",
      "location": "string",
      "startDate": "string",
      "endDate": "string",
      "responsibilities": ["string"]
    }
  ],
  "education": [
    {
      "degree": "string",
      "major": "string",
      "university": "string",
      "graduationDate": "string"
    }
  ],
  "skills": {
    "technical": ["string"],
    "soft": ["string"]
  },
  "projects": [
    {
      "projectName": "string",
      "technologies": ["string"],
      "description": "string"
    }
  ]
}
"""
        response = model.generate_content(
            [parsing_prompt, pdf_file_part],
            generation_config=genai.GenerationConfig(temperature=0)
        )
        cleaned_response = clean_json_string(response.text)
        parsed_resume_json = json.loads(cleaned_response)

        text_extraction_prompt = "Extract all text content from the provided PDF file."
        text_response = model.generate_content([text_extraction_prompt, pdf_file_part])

        return {
            "resume_text": text_response.text,
            "metadata": {"file_size_kb": len(pdf_bytes) / 1024, "filename": filename},
            "parsed_resume_json": parsed_resume_json
        }
    except Exception as e:
        logger.error(f"❌ ERROR: Failed during preprocessing or parsing with Gemini: {e}")
        raise ValueError(f"Gemini PDF Parsing Failed: {e}")

def create_scoring_runnable() -> RunnableParallel:
    """Creates a parallel runnable where each branch is a specialized scoring agent."""
    logger.info("--- Initializing Step 2: Parallel Scoring Agents with GPT-4o ---")
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    
    prompts = {
    "career_impact": """
<system_instructions>
You are the 'Career Impact Analyst'. Your role is to be an objective, evidence-based scorer.
- Your analysis MUST be based exclusively on the provided `parsed_resume_json`. Do not infer or invent information.
- Your entire output MUST be a raw JSON LIST, starting with `[` and ending with `]`.
- For each criterion, you must populate all fields in the JSON object: 'criterion', 'score', 'max_score', 'score_justification', 'suggestion', and 'wrong_thing'. Use `null` if a field is not applicable.
</system_instructions>

Score the resume's content effectiveness based on the following criteria from the provided JSON.

**Parsed Resume JSON:**
---
{parsed_resume_json}
---

**Scoring Criteria:**

- [cite_start]**Action Verbs Usage (5 pts)[cite: 4]:**
  - **Score 5:** Most bullet points in `workExperience` and `projects` start with strong, diverse action verbs (e.g., 'Orchestrated', 'Architected', 'Maximized').
  - **Score 3-4:** Good use of action verbs, but some are repetitive or weaker (e.g., 'Worked on', 'Handled').
  - **Score 1-2:** Few action verbs are used; descriptions are passive or responsibility-focused.
  - **Score 0:** No action verbs used.

- [cite_start]**Measurable Achievements (4 pts)[cite: 5]:**
  - **Score 4:** Most `responsibilities` in `workExperience` are quantified with metrics (e.g., 'Increased revenue by 30%', 'Reduced latency by 200ms').
  - **Score 2-3:** Some responsibilities are quantified, but many are purely descriptive.
  - **Score 1:** Very few or no quantifiable results. Vague claims like 'Increased efficiency'.
  - **Score 0:** No measurable achievements found.

- [cite_start]**Growth Signals (5 pts)[cite: 6]:**
  - Analyze the `workExperience` array chronologically.
  - **Score 4-5:** Clear progression shown through promotions, increasing job titles (e.g., 'Junior' to 'Senior'), or significantly expanded responsibilities between roles.
  - **Score 2-3:** Some growth is implied, but titles are flat, or responsibilities between roles are very similar.
  - **Score 1:** No clear sign of career progression. Roles are disjointed or show no increase in responsibility.
  - **Score 0:** Only one job listed or section is missing.

- [cite_start]**Skill Mentions (common hard/soft) (5 pts)[cite: 7]:**
  - **Score 4-5:** The `skills` section is well-populated with a strong mix of relevant technical and soft skills.
  - **Score 2-3:** A skills section exists but is sparse or lists only generic skills.
  - **Score 1:** Skills are mentioned only in passing within job descriptions.
  - **Score 0:** The `skills` section is null or empty.

- [cite_start]**Bullet Point Strength (6 pts)[cite: 8]:**
  - **Score 5-6:** Bullet points are consistently concise, results-oriented, and clearly state the impact of the work. Each point is a high-impact statement.
  - **Score 3-4:** Bullet points are okay but tend to be simple descriptions of duties rather than achievements.
  - **Score 1-2:** Bullet points are long, verbose, or unclear.
  - **Score 0:** No bullet points used.

- [cite_start]**Avoidance of Repetition (3 pts)[cite: 9]:**
  - **Score 3:** Descriptions and verbs used across different `workExperience` roles are unique and varied.
  - **Score 1-2:** The same phrases, responsibilities, or verbs are repeated across multiple roles, making the experience seem static.
  - **Score 0:** Content is heavily copied and pasted between roles.

- [cite_start]**Project Section Clarity (2 pts)[cite: 10]:**
  - **Score 2:** The `projects` section is present, and each project clearly lists technologies and a concise description/responsibility.
  - **Score 1:** The section exists, but project descriptions are vague, or the tech stack is unclear.
  - **Score 0:** The `projects` section is null or empty.
""",
    "layout_clarity": """
<system_instructions>
You are the 'Layout Clarity Analyst'. You are an expert at inferring visual structure from raw text.
- Your analysis MUST be based exclusively on the provided `resume_text`. IGNORE THE MEANING of the words; focus on patterns, spacing, and structure.
- Your entire output MUST be a raw JSON LIST, starting with `[` and ending with `]`.
- For each criterion, you must populate all fields in the JSON object: 'criterion', 'score', 'max_score', 'score_justification', 'suggestion', and 'wrong_thing'. Use `null` if a field is not applicable.
</system_instructions>

Score the resume's visual presentation based on inferences from the raw text provided.

**Resume Text:**
---
{resume_text}
---

**Scoring Criteria:**

- [cite_start]**Section Clarity (4 pts)[cite: 13]:**
  - **Score 4:** Sections ('Experience', 'Education', 'Skills') are clearly separated by consistent headings and ample whitespace (multiple line breaks).
  - **Score 2-3:** Sections are identifiable but have inconsistent spacing or formatting.
  - **Score 1:** Sections blend together, making the document hard to navigate.

- [cite_start]**Font & Size Compliance (3 pts)[cite: 14]:**
  - This is an inference. **Score 3** if the text looks clean and there's no evidence of unusual characters or formatting that would suggest an unprofessional font. **Score 1** if the text contains odd symbols or seems to have inconsistent character styles.

- [cite_start]**Margins & Spacing (3 pts)[cite: 15]:**
  - **Score 3:** Inferred from text flow. Lines are not excessively long, and there is consistent spacing between lines and paragraphs, suggesting balanced margins.
  - **Score 1-2:** Text seems cramped. Very long lines of text or minimal spacing between bullet points and sections.

- [cite_start]**Date & Format Consistency (4 pts)[cite: 16]:**
  - **Score 4:** All dates (e.g., 'May 2020 - Present', '05/2020 - Present') follow the exact same format throughout the document.
  - **Score 2-3:** There are minor inconsistencies (e.g., 'May 2020' in one place, 'May, 2020' in another).
  - **Score 1:** Date formats are highly inconsistent or illogical.

- [cite_start]**Readability Score (3 pts)[cite: 17]:**
  - **Score 3:** Sentences are generally short and direct. Language is clear and easy to understand.
  - **Score 1-2:** Sentences are long, complex, or contain jargon that hinders readability.

- [cite_start]**Visual Hierarchy (3 pts)[cite: 18]:**
  - **Score 3:** Job titles, company names, and section headers are consistently differentiated, likely using capitalization or spacing, making them easy to scan.
  - **Score 1-2:** It's difficult to distinguish between headings and content. All text has a similar visual weight.
""",
    "ats_compliance": """
<system_instructions>
You are the 'ATS Compliance Analyst'. You are a technical expert evaluating a resume's machine-readability.
- Your analysis MUST be based exclusively on the `resume_text` and the `parsed_resume_json`.
- Your entire output MUST be a raw JSON LIST, starting with `[` and ending with `]`.
- For each criterion, you must populate all fields in the JSON object: 'criterion', 'score', 'max_score', 'score_justification', 'suggestion', and 'wrong_thing'. Use `null` if a field is not applicable.
</system_instructions>

Score the resume's technical structure using the raw text and the parsed JSON data.

**Resume Text:**
---
{resume_text}
---
**Parsed Resume JSON:**
---
{parsed_resume_json}
---

**Scoring Criteria:**

- [cite_start]**No Tables/Columns (2 pts)[cite: 21]:**
  - **Score 2:** The `resume_text` flows in a single logical column. There are no signs of multi-column text that would confuse a parser.
  - **Score 0:** Text appears misaligned or in snippets, suggesting tables or columns were used.

- [cite_start]**No Headers/Footers (2 pts)[cite: 22]:**
  - **Score 2:** Critical contact info is present in the main body of the `parsed_resume_json`.
  - **Score 0:** Key information like name or email is missing from the parsed JSON but visible in the text, suggesting it was in a header/footer.

- [cite_start]**No Icons/Images/graph (2 pts)[cite: 23]:**
  - **Score 2:** The `resume_text` contains no non-text elements or placeholders for images (e.g., '[?]').
  - **Score 0:** The text contains characters suggesting icons (e.g., special unicode for phone/email) or images were not parsed correctly.

- [cite_start]**Parse Rate Check (Simulated) (2 pts)[cite: 24]:**
  - **Score 2:** The `parsed_resume_json` is complete and accurate. All major sections from the text (experience, education, skills) are filled.
  - **Score 1:** The JSON is mostly complete, but one or more major sections are `null` despite being present in the text.
  - **Score 0:** The JSON is mostly `null` or the extracted data is clearly incorrect/garbled.

- [cite_start]**Keyword Optimization (3 pts)[cite: 25]:**
  - **Score 3:** The text contains relevant, common keywords for a typical professional field (e.g., 'Project Management', 'Data Analysis', 'Software Development', 'Customer Relationship Management').
  - **Score 1-2:** Keywords are sparse or highly niche.
  - **Score 0:** No relevant industry keywords are found.

- [cite_start]**Section Header Standards (3 pts)[cite: 26]:**
  - **Score 3:** The `resume_text` uses standard, conventional headers like 'Experience', 'Education', 'Skills', 'Projects'.
  - **Score 1-2:** Non-standard headers are used (e.g., 'My Journey', 'What I Can Do'), which may confuse an ATS.
  - **Score 0:** No clear section headers are used.

- [cite_start]**Contact Information Placement (2 pts)[cite: 27]:**
  - **Score 2:** The `contactInformation` object in the parsed JSON is fully and correctly populated.
  - **Score 0:** Contact information is missing or incorrectly parsed.

- [cite_start]**Text Encoding Standards (2 pts)[cite: 28]:**
  - **Score 2:** Text is free of strange characters or symbols.
  - **Score 0:** Text contains special characters or symbols that could corrupt parsing.

- [cite_start]**Consistent Formatting Structure (3 pts)[cite: 29]:**
  - **Score 3:** The structure for jobs/education is consistent (e.g., Title on one line, Company on the next, Dates on the next).
  - **Score 1-2:** The formatting pattern is inconsistent across similar entries.
""",
    "linguistic_precision": """
<system_instructions>
You are the 'Linguistic Precision Analyst'. You are an expert in professional business writing and grammar.
- Your analysis MUST be based exclusively on the provided `resume_text`.
- Your entire output MUST be a raw JSON LIST, starting with `[` and ending with `]`.
- For each criterion, you must populate all fields in the JSON object: 'criterion', 'score', 'max_score', 'score_justification', 'suggestion', and 'wrong_thing'. Use `null` if a field is not applicable.
</system_instructions>

Score the resume's writing style, tone, and clarity based on the raw text.

**Resume Text:**
---
{resume_text}
---

**Scoring Criteria:**

- [cite_start]**Spelling & Grammar (5 pts)[cite: 32]:**
  - **Score 5:** No spelling or grammatical errors found.
  - **Score 1-4:** Deduct points for each error found.
  - **Score 0:** Pervasive errors throughout the document.

- [cite_start]**Active Voice Usage (4 pts)[cite: 33]:**
  - **Score 4:** The vast majority of statements are in the active voice (e.g., 'Managed a team of 5').
  - **Score 1-3:** Significant use of passive voice (e.g., 'A team of 5 was managed by me').
  - **Score 0:** Overwhelmingly passive language.

- [cite_start]**Weak Verb Avoidance (3 pts)[cite: 34]:**
  - **Score 3:** Avoids weak or vague verbs. Descriptions are forceful and specific.
  - **Score 1-2:** Frequent use of weak verbs like 'Assisted with', 'Helped', 'Was responsible for'.
  - **Score 0:** Most verbs are weak and non-descriptive.

- [cite_start]**Buzzword Avoidance (3 pts)[cite: 35]:**
  - **Score 3:** Language is professional and direct, avoiding empty jargon (e.g., 'synergy', 'go-getter', 'results-driven').
  - **Score 1-2:** Some use of clichés or buzzwords that add little value.
  - **Score 0:** Over-reliance on buzzwords.

- [cite_start]**Personal Pronouns Avoided (3 pts)[cite: 36]:**
  - **Score 3:** No use of personal pronouns ('I', 'me', 'my').
  - **Score 0:** Any use of personal pronouns is detected.

- [cite_start]**Line-by-Line Clarity (2 pts)[cite: 37]:**
  - **Score 2:** Each sentence is clear, unambiguous, and easy to understand on the first read.
  - **Score 1:** Some sentences are convoluted, requiring re-reading.
  - **Score 0:** Many sentences are unclear or poorly constructed.

- [cite_start]**Sentence Flow & Coherence (2 pts)[cite: 38]:**
  - **Score 2:** The ideas flow logically; the document is coherent and easy to follow.
  - **Score 1:** The flow is choppy or ideas are presented in a disorganized manner.

- [cite_start]**Tense Consistency (2 pts)[cite: 39]:**
  - **Score 2:** Correctly uses past tense for all previous roles and present tense for the current role.
  - **Score 0:** Inconsistent tense usage (e.g., using present tense for a past job).
""",
    "document_standards": """
<system_instructions>
You are the 'Document Standards Analyst'. Your task is to score the resume's core properties based on metadata and text.
- Your analysis MUST be based exclusively on the provided `resume_text` and `metadata`.
- Your entire output MUST be a raw JSON LIST, starting with `[` and ending with `]`.
- For each criterion, you must populate all fields in the JSON object: 'criterion', 'score', 'max_score', 'score_justification', 'suggestion', and 'wrong_thing'. Use `null` if a field is not applicable.
</system_instructions>

Score the resume's structural integrity using the provided text and metadata.

**Resume Text (for word count):**
---
{resume_text}
---
**Metadata (for file properties):**
---
{metadata}
---

**Scoring Criteria:**

- [cite_start]**File Type Validity (1 pt)[cite: 42]:**
  - Check the file type from the metadata.
  - **Score 1:** File type is .docx or .pdf.
  - **Score 0:** File type is not .docx or .pdf.

- [cite_start]**File Size (1 pt)[cite: 43]:**
  - Check the file size from the metadata.
  - **Score 1:** File size is under 1MB.
  - **Score 0:** File size is 1MB or greater.

- [cite_start]**Page Count (1 pt)[cite: 44]:**
  - Infer from word count. Assume ~450 words per page.
  - **Score 1:** Estimated page count is 1-2 pages.
  - **Score 0:** Estimated page count is greater than 2 pages.

- [cite_start]**Word Count Range (2 pts)[cite: 45]:**
  - Count the words in `resume_text`.
  - **Score 2:** Word count is between 250 and 750.
  - **Score 1:** Word count is between 150-249 or 751-850.
  - **Score 0:** Word count is outside the 150-850 range.
"""
}

    scoring_chains = {}
    for name, prompt_text in prompts.items():
        prompt = ChatPromptTemplate.from_template(template=prompt_text)
        chain = prompt | llm | StrOutputParser() | RunnableLambda(parse_llm_json_list)
        scoring_chains[name] = chain
    
    return RunnableParallel(**scoring_chains)

def assemble_final_report(chain_result: Dict) -> Dict:
    """Takes the output of the parallel scorer and the initial parsed data to build the final report."""
    logger.info("--- Running Step 3: Assembling Final JSON Report ---")
    max_scores = {
        "document_standards": 5, "career_impact": 30, "layout_clarity": 20, 
        "ats_compliance": 21, "linguistic_precision": 24
    }
    
    parsed_data = chain_result['parsed_data']
    scores_by_category = chain_result['scores']

    detailed_breakdown = {}
    category_scores_map = {}
    total_score = 0

    for category, score_list in scores_by_category.items():
        validated_scores = [DetailedScore(**item) for item in score_list]
        detailed_breakdown[category] = validated_scores

        category_score = sum(detail.score for detail in validated_scores)
        total_score += category_score
        category_scores_map[category] = f"{category_score}/{max_scores.get(category, 0)}"

    final_report_data = {
        "total_score": f"{total_score}/100",
        "category_scores": category_scores_map,
        "detailed_breakdown": detailed_breakdown,
        "parsed_resume_content": parsed_data['parsed_resume_json']
    }
    return final_report_data