-- Raw Data Tables
CREATE TABLE multipliers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    value FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    session_id TEXT
);

CREATE TABLE crash_values (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    value FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    session_id TEXT
);

-- Model Persistence Tables
CREATE TABLE model_versions (
    id SERIAL PRIMARY KEY,
    version TEXT NOT NULL UNIQUE,
    model_weights BYTEA NOT NULL,
    accuracy FLOAT DEFAULT 75.0,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE training_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version TEXT REFERENCES model_versions(version),
    samples_used INT,
    accuracy_before FLOAT,
    accuracy_after FLOAT,
    training_duration INTERVAL,
    loss FLOAT,
    timestamp TIMESTAMP DEFAULT NOW()
);

CREATE TABLE accuracy_tracking (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version TEXT REFERENCES model_versions(version),
    timestamp TIMESTAMP DEFAULT NOW(),
    accuracy FLOAT
);