CREATE TABLE IF NOT EXISTS articles (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    published TIMESTAMP NOT NULL,
    link TEXT UNIQUE NOT NULL,
    image BYTEA,
    tags TEXT[],
    summary TEXT
);
