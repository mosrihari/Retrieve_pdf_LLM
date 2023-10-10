import os
from chromadb.config import Settings

CHROME_DB_SETTINGS = Settings(
    chroma_db_impl = 'duckdb+parquet',
    persist_directory = 'db',
    anonymized_telemetry = False
)