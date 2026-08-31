CREATE TABLE IF NOT EXISTS psychology_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL UNIQUE REFERENCES users(id),
    dominant_pattern VARCHAR(30),
    lz_complexity_score FLOAT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
