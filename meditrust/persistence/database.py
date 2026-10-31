import sqlite3
from pathlib import Path

SCHEMA = '''
CREATE TABLE IF NOT EXISTS hospitals (id TEXT PRIMARY KEY, name TEXT, dataset_size INTEGER, trust REAL DEFAULT 80, last_round INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS validation_runs (id INTEGER PRIMARY KEY, hospital_id TEXT, round_id INTEGER, total INTEGER, valid INTEGER, review INTEGER, rejected INTEGER, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS rounds (id INTEGER PRIMARY KEY, model_version INTEGER, status TEXT, contributors TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS decisions (id INTEGER PRIMARY KEY, round_id INTEGER, hospital_id TEXT, decision TEXT, norm REAL, cosine REAL, score REAL, explanation TEXT, attack TEXT, review TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS metrics (id INTEGER PRIMARY KEY, round_id INTEGER, model_version INTEGER, accuracy REAL, precision REAL, recall REAL, f1 REAL, status TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS approved_models (model_version INTEGER PRIMARY KEY, round_id INTEGER, params TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS trust_history (id INTEGER PRIMARY KEY, hospital_id TEXT, score REAL, reason TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS audit_events (id INTEGER PRIMARY KEY, event_type TEXT, hospital_id TEXT, round_id INTEGER, decision TEXT, explanation TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
'''
def connect(path):
    con=sqlite3.connect(Path(path)); con.row_factory=sqlite3.Row; con.executescript(SCHEMA); return con
