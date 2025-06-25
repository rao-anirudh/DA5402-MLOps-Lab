def insert_into_postgres(**kwargs):

    from airflow.providers.postgres.hooks.postgres import PostgresHook
    import base64
    import os

    ti = kwargs["ti"]
    scraped_data = ti.xcom_pull(task_ids="scrape_google_news")
    if not scraped_data:
        count = 0
    else:
        pg_hook = PostgresHook(postgres_conn_id="tutorial_pg_conn")
        conn = pg_hook.get_conn()
        cursor = conn.cursor()
        insert_query = """
            INSERT INTO news (article_date, headline, newspaper, scrape_time, thumbnail, url)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (headline) DO NOTHING;
        """
        count = 0
        for record in scraped_data:
            thumbnail_data = base64.b64decode(record["thumbnail"]) if record["thumbnail"] else None
            cursor.execute(insert_query, (
                record["article_date"],
                record["headline"],
                record["newspaper"],
                record["scrape_time"],
                thumbnail_data,
                record.get("url")
            ))
            if cursor.rowcount > 0:
                count += 1
        conn.commit()
        cursor.close()
        conn.close()

    status_file_path = "/tmp/dags/run/status"
    os.makedirs(os.path.dirname(status_file_path), exist_ok=True)

    with open(status_file_path, "w") as f:
        f.write(str(count))
