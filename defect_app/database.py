"""SQLite storage for the inspection history."""
import sqlite3
from datetime import datetime


class InspectionDB:
    def __init__(self, path):
        self.path = path
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS inspections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,          -- 'upload' or 'live'
                    filename TEXT,
                    label TEXT NOT NULL,           -- 'OK' or 'DEFECTIVE'
                    defect_prob REAL NOT NULL,
                    confidence REAL NOT NULL,
                    inference_ms REAL NOT NULL,
                    image_file TEXT,
                    heatmap_file TEXT
                )""")

    def _connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def log(self, source, filename, result, image_file=None, heatmap_file=None):
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO inspections (timestamp, source, filename, label, defect_prob, confidence,
                                            inference_ms, image_file, heatmap_file)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), source, filename, result['label'],
                 result['defect_prob'], result['confidence'], result['inference_ms'], image_file, heatmap_file))
            return cur.lastrowid

    def stats(self):
        with self._connect() as conn:
            row = conn.execute("""
                SELECT COUNT(*) AS total,
                       COALESCE(SUM(label = 'DEFECTIVE'), 0) AS defective,
                       COALESCE(AVG(confidence), 0) AS avg_confidence,
                       COALESCE(AVG(inference_ms), 0) AS avg_ms
                FROM inspections""").fetchone()
            daily = conn.execute("""
                SELECT substr(timestamp, 1, 10) AS day,
                       SUM(label = 'OK') AS ok, SUM(label = 'DEFECTIVE') AS defective
                FROM inspections GROUP BY day ORDER BY day DESC LIMIT 14""").fetchall()
        stats = dict(row)
        stats['ok'] = stats['total'] - stats['defective']
        stats['defect_rate'] = 100 * stats['defective'] / stats['total'] if stats['total'] else 0
        stats['daily'] = [dict(r) for r in reversed(daily)]
        return stats

    def recent(self, limit=50):
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(
                'SELECT * FROM inspections ORDER BY id DESC LIMIT ?', (limit,)).fetchall()]

    def all_rows(self):
        with self._connect() as conn:
            return [dict(r) for r in conn.execute('SELECT * FROM inspections ORDER BY id').fetchall()]
