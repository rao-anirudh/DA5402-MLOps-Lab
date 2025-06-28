This periodically fetches articles from an RSS feed, stores them in a PostgreSQL database, and serves a web interface to browse the articles published today.

All components are containerised and orchestrated using `docker-compose`.

To build and launch, run:

```bash
docker-compose down -v       
docker-compose up --build    
```

Wait for logs to show the database is healthy and the RSS reader is inserting entries.

Visit:

```
http://localhost:8043
```

You will see articles from **today**. Each entry includes the title, publication time, image (if available), and summary.
