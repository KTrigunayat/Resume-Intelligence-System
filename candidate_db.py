import sqlite3
import json
import os
from typing import List, Dict, Any

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'output', 'candidates.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT,
            candidate_name TEXT,
            score REAL,
            analysis_json TEXT
        )
    ''')
    conn.commit()
    conn.close()

def insert_candidate_analysis(job_id: str, candidate_name: str, score: float, analysis: Dict[str, Any]):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO candidates (job_id, candidate_name, score, analysis_json)
        VALUES (?, ?, ?, ?)
    ''', (job_id, candidate_name, score, json.dumps(analysis)))
    conn.commit()
    conn.close()

def get_top_candidates(job_id: str, top_n: int = 10) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT candidate_name, score, analysis_json FROM candidates
        WHERE job_id = ?
        ORDER BY score DESC
        LIMIT ?
    ''', (job_id, top_n))
    rows = c.fetchall()
    conn.close()
    results = []
    for name, score, analysis_json in rows:
        analysis = json.loads(analysis_json)
        results.append({
            'candidate_name': name,
            'score': score,
            'analysis': analysis
        })
    return results

def clear_candidates_for_job(job_id: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM candidates WHERE job_id = ?', (job_id,))
    conn.commit()
    conn.close() 