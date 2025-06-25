from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime
import os
from email_sender import send_email

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 2, 11),
}

email_dag = DAG(
    "email_dag",
    default_args=default_args,
    description="Sends email notification when new headlines appear",
    schedule_interval=None,
)

file_sensor_task = FileSensor(
    task_id="wait_for_status_file",
    filepath="/tmp/dags/run/status",
    poke_interval=60,
    timeout=3660,
    dag=email_dag,
)


def check_new_records():

    from airflow.providers.postgres.hooks.postgres import PostgresHook
    import os
    import json

    pg_hook = PostgresHook(postgres_conn_id="tutorial_pg_conn")
    conn = pg_hook.get_conn()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM news;")
    current_count = cursor.fetchone()[0]
    cursor.close()
    conn.close()

    state_file = "/tmp/previous_record_count.json"
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            previous_count = json.load(f).get("record_count", 0)
    else:
        previous_count = 0

    new_entries = current_count - previous_count

    with open(state_file, "w") as f:
        json.dump({"record_count": current_count}, f)

    return new_entries if new_entries > 0 else None


check_new_records_task = PythonOperator(
    task_id="check_new_records",
    python_callable=check_new_records,
    provide_context=True,
    dag=email_dag,
)

email_task = PythonOperator(
    task_id="send_email",
    python_callable=send_email,
    provide_context=True,
    dag=email_dag,
)

delete_status_file_task = PythonOperator(
    task_id="delete_status_file",
    python_callable=lambda: os.remove("/tmp/dags/run/status"),
    dag=email_dag,
)

self_trigger = TriggerDagRunOperator(
    task_id="self_trigger",
    trigger_dag_id="email_dag",
    wait_for_completion=False,
    dag=email_dag,
)

file_sensor_task >> check_new_records_task >> email_task >> delete_status_file_task >> self_trigger
