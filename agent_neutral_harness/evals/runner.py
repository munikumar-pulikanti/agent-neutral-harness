"""Golden baseline eval runner with LLM-as-a-judge regression scoring."""

import sqlite3
import requests
from typing import List, Dict, Any


def run_eval_judge(task: str, expected_answer: str, actual_answer: str, judge_model: str = "llama3.1:8b", base_url: str = "http://localhost:11434") -> Dict[str, Any]:
    prompt = f"""You are an impartial AI evaluator.
Task: {task}
Golden Expected Answer: {expected_answer}
Actual Agent Response: {actual_answer}

Did the actual agent response fulfill the task accurately according to the golden baseline?
Respond with JSON: {{"passed": true/false, "reasoning": "brief explanation"}}"""

    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json={"model": judge_model, "prompt": prompt, "format": "json", "stream": False, "options": {"temperature": 0}},
            timeout=30,
        ).json()
        return resp.get("response", {})
    except Exception as e:
        return {"passed": False, "reasoning": f"Eval judge error: {str(e)}"}