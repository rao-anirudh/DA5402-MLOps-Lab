import os, time, feedparser, psycopg2, requests
from datetime import datetime
from psycopg2.extras import execute_values

DB_PARAMS = {
    "dbname": os.environ["POSTGRES_DB"],
    "user": os.environ["POSTGRES_USER"],
    "password": os.environ["POSTGRES_PASSWORD"],
    "host": os.environ["DB_HOST"],
    "port": os.environ["DB_PORT"],
}
RSS_URL = os.environ["RSS_URL"]
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", 300))


def get_image(entry):
    try:
        media = entry.media_content[0]["url"]
        return requests.get(media).content
    except Exception:
        return None


def insert_articles(entries):
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    records = []
    for e in entries:
        try:
            published = datetime(*e.published_parsed[:6])
            image_data = get_image(e)
            records.append((
                e.title, published, e.link, image_data,
                [t.term for t in e.tags] if hasattr(e, "tags") else [],
                e.summary
            ))
        except Exception as ex:
            print(f"Skipping entry: {ex}")
    if records:
        execute_values(cur,
                       "INSERT INTO articles (title, published, link, image, tags, summary) VALUES %s ON CONFLICT DO NOTHING",
                       records)
    conn.commit()
    cur.close()
    conn.close()


while True:
    feed = feedparser.parse(RSS_URL)
    insert_articles(feed.entries)
    time.sleep(POLL_INTERVAL)
