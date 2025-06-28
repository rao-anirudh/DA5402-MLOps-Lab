import os
from flask import Flask, render_template, send_file, abort
import psycopg2
from io import BytesIO
from datetime import date

app = Flask(__name__)
conn = psycopg2.connect(
    dbname=os.environ["POSTGRES_DB"],
    user=os.environ["POSTGRES_USER"],
    password=os.environ["POSTGRES_PASSWORD"],
    host=os.environ["DB_HOST"],
    port=os.environ["DB_PORT"]
)


@app.route('/')
def index():
    cur = conn.cursor()
    cur.execute(
        "SELECT id, title, published, link, summary FROM articles WHERE published::date = %s ORDER BY published DESC",
        (date.today(),))
    articles = cur.fetchall()
    cur.close()
    return render_template('index.html', articles=articles)


@app.route('/image/<int:id>')
def image(id):
    cur = conn.cursor()
    cur.execute("SELECT image FROM articles WHERE id = %s", (id,))
    row = cur.fetchone()
    cur.close()
    if row and row[0]:
        return send_file(BytesIO(row[0]), mimetype='image/jpeg')
    return abort(404)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
