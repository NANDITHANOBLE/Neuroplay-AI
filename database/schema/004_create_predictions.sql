CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    move_id INTEGER NOT NULL REFERENCES moves(id),
    model_name VARCHAR(30) NOT NULL,
    predicted_move SMALLINT NOT NULL CHECK (predicted_move IN (0,1,2)),
    confidence FLOAT NOT NULL,
    explanation_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_predictions_move_id ON predictions(move_id);
