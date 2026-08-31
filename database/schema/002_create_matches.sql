CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    mode VARCHAR(20) NOT NULL DEFAULT 'keyboard',
    final_score_user INTEGER DEFAULT 0,
    final_score_ai INTEGER DEFAULT 0,
    model_used VARCHAR(30) DEFAULT 'markov'
);
