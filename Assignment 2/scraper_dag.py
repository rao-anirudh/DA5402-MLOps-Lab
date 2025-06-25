from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.postgres_operator import PostgresOperator
from datetime import datetime
import os
from google_news_scraper import scrape_google_news
from data_manager import insert_into_postgres

config = {
  "url": "https://news.google.com/home/",
  "load_time": 5,
  "top_stories_string": "Top stories",
  "article_class": "article.IBr9hb",
  "headline_class": "article.IBr9hb > a.gPFEn",
  "thumbnail_class": "figure.K0q4G.P22Vib > img.Quavad.vwBmvb",
  "newspaper_class": "article.IBr9hb > div.MCAGUe",
  "date_class": "article.IBr9hb > div.UOVeFe > time.hvbAAd",
}

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 2, 11),
}

scraper_dag = DAG(
    "scraper_dag",
    default_args=default_args,
    description="Scrapes Google News and stores data in PostgreSQL",
    schedule_interval="0 * * * *",
)

install_dependencies = BashOperator(
    task_id="install_dependencies",
    bash_command="pip install selenium requests pymongo",
    dag=scraper_dag,
)

scrape = PythonOperator(
    task_id="scrape_google_news",
    python_callable=scrape_google_news,
    op_kwargs={"config": config},
    dag=scraper_dag,
)

setup_postgres_table = PostgresOperator(
    task_id="setup_postgres_table",
    postgres_conn_id="tutorial_pg_conn",
    sql="""
        CREATE TABLE IF NOT EXISTS news (
            id SERIAL PRIMARY KEY,
            article_date DATE,
            headline TEXT UNIQUE,
            newspaper TEXT,
            scrape_time TIMESTAMP,
            thumbnail BYTEA,
            url TEXT
        );
    """,
    dag=scraper_dag,
)

insert_db_task = PythonOperator(
    task_id="insert_into_postgres",
    python_callable=insert_into_postgres,
    provide_context=True,
    dag=scraper_dag,
)

install_dependencies >> scrape >> setup_postgres_table >> insert_db_task
