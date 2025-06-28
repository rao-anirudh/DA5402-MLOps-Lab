FROM postgres:15
ENV POSTGRES_DB=admin \
    POSTGRES_USER=admin \
    POSTGRES_PASSWORD=admin
COPY init-db.sql /docker-entrypoint-initdb.d/
