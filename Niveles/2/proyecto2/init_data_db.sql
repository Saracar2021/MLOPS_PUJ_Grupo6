CREATE TABLE IF NOT EXISTS batch_data (
    id SERIAL PRIMARY KEY,
    group_number INTEGER NOT NULL,
    batch_number INTEGER NOT NULL,
    execution_date TIMESTAMP NOT NULL,
    data_json JSONB NOT NULL,
    row_count INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_batch_data_group ON batch_data(group_number);
CREATE INDEX idx_batch_data_execution ON batch_data(execution_date);
