CREATE TABLE IF NOT EXISTS drift_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL REFERENCES matches(id),
    detected_at_round INTEGER NOT NULL,
    detector_method VARCHAR(20) NOT NULL,
    severity VARCHAR(10) NOT NULL CHECK (severity IN ('warning','drift')),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
