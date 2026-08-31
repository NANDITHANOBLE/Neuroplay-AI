CREATE TABLE IF NOT EXISTS moves (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL REFERENCES matches(id),
    round_number INTEGER NOT NULL,
    player_move SMALLINT NOT NULL CHECK (player_move IN (0,1,2)),
    ai_move SMALLINT NOT NULL CHECK (ai_move IN (0,1,2)),
    result VARCHAR(10) NOT NULL CHECK (result IN ('win','loss','draw')),
    reaction_time_ms INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_moves_match_id ON moves(match_id);
CREATE INDEX IF NOT EXISTS idx_moves_timestamp ON moves(timestamp);
