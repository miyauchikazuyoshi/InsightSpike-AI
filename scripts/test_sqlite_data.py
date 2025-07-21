#!/usr/bin/env python3
"""Test SQLite data access without async issues"""

import sqlite3
from pathlib import Path

# Create test database
db_path = Path("./data/sqlite/insightspike.db")
db_path.parent.mkdir(parents=True, exist_ok=True)

# Connect and create tables
conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# Create episodes table
cursor.execute("""
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    text TEXT NOT NULL,
    vector BLOB,
    c_value REAL DEFAULT 0.5,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

# Create indices
cursor.execute("CREATE INDEX IF NOT EXISTS idx_episodes_namespace ON episodes(namespace)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_episodes_created ON episodes(created_at)")

# Insert test data
import numpy as np

test_episodes = [
    ("test1", "default", "Machine learning involves pattern recognition.", 0.5),
    ("test2", "default", "Neural networks mimic brain structure.", 0.6),
    ("test3", "default", "Deep learning uses multiple layers.", 0.7),
]

for ep_id, namespace, text, c_value in test_episodes:
    # Create dummy vector
    vec = np.random.rand(384).astype(np.float32)
    vec_bytes = vec.tobytes()
    
    cursor.execute("""
    INSERT OR REPLACE INTO episodes (id, namespace, text, vector, c_value, metadata)
    VALUES (?, ?, ?, ?, ?, '{}')
    """, (ep_id, namespace, text, vec_bytes, c_value))

conn.commit()

# Query data
cursor.execute("SELECT COUNT(*) FROM episodes")
count = cursor.fetchone()[0]
print(f"Total episodes: {count}")

cursor.execute("SELECT id, text, c_value FROM episodes")
for row in cursor.fetchall():
    print(f"  - {row[0]}: {row[1][:50]}... (c={row[2]})")

conn.close()
print("\nSQLite database initialized successfully!")