import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from typing import Optional

class SQLHandler:
    """Simplified SQL handler for document chunk retrieval."""

    def __init__(self, config, connection_string, engine: Optional[Engine] = None):
        self.config = config
        self.connection_string = connection_string
        self.engine: Engine = engine or create_engine(connection_string)

    def get_chunks_by_document_id(self, document_id: str) -> pd.DataFrame:
        """Retrieves all chunks for a given document ID."""
        if not document_id:
            return pd.DataFrame()

        try:
            with self.engine.connect() as conn:
                query = text("""
                    SELECT chunk_id, content, tokenized_content, metadata
                    FROM document_chunks
                    WHERE document_id = :doc_id AND collection_name = :coll_name
                    ORDER BY
                        COALESCE(
                            CAST(SUBSTRING(chunk_id FROM '_chunk_(\\d+)') AS INTEGER),
                            999999
                        ),
                        chunk_id
                """)
                df = pd.read_sql(query, conn, params={"doc_id": document_id, "coll_name": self.config.collection_name})
            return df
        except Exception as e:
            print(f"Error getting chunks for document {document_id}: {e}")
            return pd.DataFrame()
