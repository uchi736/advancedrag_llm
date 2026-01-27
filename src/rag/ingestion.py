"""
Document ingestion handler with improved batch processing and connection management.
"""
import json
import re
from pathlib import Path
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.schema import Document
from src.rag.text_processor import JapaneseTextProcessor
from sqlalchemy import text


class IngestionHandler:
    def __init__(self, config, vector_store, engine, text_processor=None):
        self.config = config
        self.vector_store = vector_store
        self.engine = engine
        self.text_processor = text_processor if text_processor else JapaneseTextProcessor()

        # Initialize Azure Document Intelligence processor
        from src.rag.pdf_processors.azure_di_processor import AzureDocumentIntelligenceProcessor
        self.pdf_processor = AzureDocumentIntelligenceProcessor(config)
        if not self.pdf_processor:
            print("Warning: Azure Document Intelligence processor could not be initialized")

    def load_documents(self, paths: List[str]) -> List[Document]:
        all_docs = []
        for p_str in paths:
            path = Path(p_str)
            if not path.exists():
                print(f"Path {path} does not exist.")
                continue

            # Load documents based on file type
            docs = self._load_single_document(str(path))
            all_docs.extend(docs)

        return all_docs

    def _load_single_document(self, path: str) -> List[Document]:
        """Load a single PDF document using Azure Document Intelligence."""
        docs = []
        p = Path(path)

        if p.suffix.lower() != '.pdf':
            print(f"Skipping non-PDF file: {path}")
            return docs

        try:
            docs = self.pdf_processor.process(path)
        except Exception as e:
            print(f"Error loading {path}: {type(e).__name__} - {e}")

        return docs

    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        """Chunk documents into smaller pieces."""
        return self._chunk_documents_standard(docs)

    def _chunk_documents_standard(self, docs: List[Document]) -> List[Document]:
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

        # Fallback splitter for content without headers
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )

        def normalize_header_line(line: str) -> str:
            """Normalize Markdown header spacing for comparison."""
            if not line:
                return ""
            match = re.match(r'^(#{1,6})\s*(.*)$', line.strip())
            if not match:
                return line.strip()
            marker, title = match.groups()
            return f"{marker} {title}".strip()

        def ensure_header(text: str, header_line: str) -> str:
            """Ensure the chunk starts with the desired header line at most once."""
            if not text:
                return header_line or ""

            stripped = text.lstrip('\n')
            if not header_line:
                return stripped

            normalized_header = normalize_header_line(header_line)
            lines = stripped.splitlines()
            if lines:
                first_line_normalized = normalize_header_line(lines[0])
                if first_line_normalized == normalized_header:
                    return stripped

            return f"{header_line}\n\n{stripped}" if stripped else header_line

        def remove_leading_header(text: str, header_line: str) -> str:
            """Strip the leading header line (if present) from a chunk."""
            if not text:
                return ""

            stripped = text.lstrip('\n')
            if not header_line:
                return stripped

            normalized_header = normalize_header_line(header_line)
            lines = stripped.splitlines()
            if lines:
                first_line_normalized = normalize_header_line(lines[0])
                if first_line_normalized == normalized_header:
                    remainder = "\n".join(lines[1:])
                    return remainder.lstrip('\n')

            return stripped

        def has_meaningful_body(text: str, header_line: str) -> bool:
            """Return True when a chunk has content beyond an optional leading header."""
            body = remove_leading_header(text, header_line)
            return bool(body.strip())

        all_chunks = []
        for i, d in enumerate(docs):
            src = d.metadata.get("source", f"doc_source_{i}")
            doc_id = Path(src).name
            chunk_counter = 0  # Reset counter for each document

            try:
                # Normalize text content first
                normalized_content = self.text_processor.normalize_text(d.page_content)

                # Ensure markdown headers are properly formatted for splitting
                # Add newline before headers if missing
                normalized_content = re.sub(r'([^\n])(#{1,6})', r'\1\n\2', normalized_content)
                # Add space after header markers if missing
                normalized_content = re.sub(r'(^#{1,6})([^# \n])', r'\1 \2', normalized_content, flags=re.MULTILINE)

                # Split by markdown headers
                md_header_splits = markdown_splitter.split_text(normalized_content)

                # Debug logging (with encoding safety for Windows)
                print(f"Markdown splitter created {len(md_header_splits)} sections from {doc_id}")
                for idx, split in enumerate(md_header_splits[:3]):  # Show first 3 sections
                    try:
                        preview = split.page_content[:100].replace('\n', ' ')
                        # Try to safely encode for Windows console
                        safe_preview = preview.encode('cp932', errors='replace').decode('cp932')
                        print(f"  Section {idx}: {safe_preview}...")
                    except Exception:
                        # If encoding fails, show a generic message
                        print(f"  Section {idx}: [Content preview unavailable due to encoding]")

                # Further split chunks that are too large
                doc_chunks = []
                for split in md_header_splits:
                    # Restore headers from metadata to content
                    content = split.page_content
                    header_line = ""

                    # Add headers back to the beginning of content
                    if split.metadata:
                        for marker, meta_key in headers_to_split_on:
                            value = split.metadata.get(meta_key)
                            if value and isinstance(value, str) and value.strip():
                                header_line = f"{marker} {value.strip()}"

                    # Update the page_content with headers included
                    split.page_content = ensure_header(content, header_line)

                    # Now split if too large
                    if len(split.page_content) > self.config.chunk_size:
                        sub_chunks = text_splitter.split_documents([split])
                        for sub_chunk in sub_chunks:
                            # Ensure metadata from header split is preserved
                            sub_chunk.metadata.update(split.metadata)
                            sub_chunk.page_content = ensure_header(sub_chunk.page_content, header_line)

                            if not has_meaningful_body(sub_chunk.page_content, header_line):
                                continue

                            doc_chunks.append(sub_chunk)
                    else:
                        if has_meaningful_body(split.page_content, header_line):
                            doc_chunks.append(split)

                # Add metadata to chunks for this document
                for chunk in doc_chunks:
                    chunk.metadata["chunk_id"] = f"{doc_id}_chunk_{chunk_counter}"
                    chunk.metadata["document_id"] = doc_id
                    chunk.metadata["source"] = src
                    chunk.metadata["chunk_index"] = chunk_counter
                    chunk.metadata["is_parent"] = False
                    chunk.metadata["type"] = "document"
                    chunk_counter += 1
                    all_chunks.append(chunk)

            except Exception as e:
                try:
                    error_msg = f"Error chunking document {doc_id}: {type(e).__name__}: {str(e)}"
                    print(error_msg.encode('cp932', errors='replace').decode('cp932'))
                except:
                    print(f"Error chunking document: [encoding error in error message]")

                try:
                    import traceback
                    traceback.print_exc()
                except:
                    print("Could not print stack trace due to encoding issues")
                continue

        return all_chunks

    def _store_chunks_for_keyword_search(self, chunks: List[Document]):
        """Store chunks in PostgreSQL for keyword search."""
        if not chunks or not self.engine:
            return

        sql = text("""
            INSERT INTO document_chunks(collection_name, document_id, chunk_id, content, tokenized_content, metadata, created_at)
            VALUES(:coll_name, :doc_id, :cid, :cont, :tok_cont, :meta, CURRENT_TIMESTAMP)
            ON CONFLICT(chunk_id) DO UPDATE SET
                content = EXCLUDED.content, tokenized_content = EXCLUDED.tokenized_content,
                metadata = EXCLUDED.metadata, document_id = EXCLUDED.document_id,
                collection_name = EXCLUDED.collection_name, created_at = CURRENT_TIMESTAMP;
        """)

        try:
            with self.engine.connect() as conn, conn.begin():
                for c in chunks:
                    normalized_content = self.text_processor.normalize_text(c.page_content)
                    tokenized_content = self.text_processor.tokenize(normalized_content) if self.config.enable_japanese_search else ""
                    conn.execute(sql, {
                        "coll_name": self.config.collection_name,
                        "doc_id": c.metadata["document_id"],
                        "cid": c.metadata["chunk_id"],
                        "cont": normalized_content,
                        "tok_cont": " ".join(tokenized_content),
                        "meta": json.dumps(c.metadata or {})
                    })
        except Exception as e:
            print(f"Error storing chunks for keyword search: {type(e).__name__} - {e}")

    def ingest_documents(self, paths: List[str]):
        """Ingest documents with improved batch processing and error handling."""
        BATCH_SIZE = 50  # Process in smaller batches to avoid connection timeout
        total_chunks_processed = 0
        failed_files = []
        successful_files = []

        for path_idx, path in enumerate(paths):
            try:
                print(f"\nProcessing file {path_idx + 1}/{len(paths)}: {path}")

                # Load single document
                docs = self._load_single_document(path)
                if not docs:
                    print(f"No documents loaded from {path}")
                    continue

                # Chunk the document
                print(f"Chunking {len(docs)} document(s) from {path}...")
                chunks = self.chunk_documents(docs)

                # Filter valid chunks
                valid_chunks = [c for c in chunks if c.page_content and c.page_content.strip()]

                if not valid_chunks:
                    print(f"No valid chunks from {path}")
                    continue

                # Process in batches
                print(f"Ingesting {len(valid_chunks)} chunks in batches of {BATCH_SIZE}...")
                file_processed = False

                for i in range(0, len(valid_chunks), BATCH_SIZE):
                    batch = valid_chunks[i:i+BATCH_SIZE]
                    batch_ids = [c.metadata['chunk_id'] for c in batch]

                    try:
                        # Store batch in vector store
                        self.vector_store.add_documents(batch, ids=batch_ids)

                        # Store batch for keyword search
                        self._store_chunks_for_keyword_search(batch)

                        batch_end = min(i + BATCH_SIZE, len(valid_chunks))
                        print(f"  Processed chunks {i+1}-{batch_end}/{len(valid_chunks)}")
                        total_chunks_processed += len(batch)
                        file_processed = True

                    except Exception as batch_error:
                        print(f"  Error processing batch {i//BATCH_SIZE + 1}: {batch_error}")
                        # Try to reconnect for next batch
                        if self.engine:
                            try:
                                self.engine.dispose()  # Close all connections
                                print("  Reconnecting to database...")
                            except:
                                pass
                        continue

                if file_processed:
                    successful_files.append(path)
                    print(f"✓ Successfully processed {path}")
                else:
                    failed_files.append(path)
                    print(f"✗ Failed to process {path}")

            except Exception as e:
                print(f"Error processing {path}: {type(e).__name__} - {e}")
                failed_files.append(path)
                continue  # Continue with next file

        # Summary
        print(f"\n{'='*60}")
        print(f"Ingestion complete:")
        print(f"  Total chunks processed: {total_chunks_processed}")
        print(f"  Successful files: {len(successful_files)}/{len(paths)}")
        if failed_files:
            print(f"  Failed files ({len(failed_files)}):")
            for f in failed_files:
                print(f"    - {f}")
        print(f"{'='*60}")

    def delete_document_by_id(self, doc_id: str, collection_name: str = None) -> tuple[bool, str]:
        """Delete a document and all its chunks from a specific collection.

        Args:
            doc_id: Document ID to delete
            collection_name: Collection name to delete from (defaults to self.config.collection_name)

        Returns:
            Tuple of (success: bool, message: str)
        """
        if not doc_id:
            return False, "Document ID cannot be empty."

        # Use provided collection_name or fall back to config
        coll_name = collection_name or self.config.collection_name

        try:
            if not self.engine:
                return False, "Database engine not available."

            with self.engine.connect() as conn:
                # 1) 取得: 対象ドキュメントのチャンクID
                result = conn.execute(
                    text(
                        "SELECT chunk_id FROM document_chunks "
                        "WHERE document_id = :doc_id AND collection_name = :coll_name"
                    ),
                    {"doc_id": doc_id, "coll_name": coll_name}
                )
                chunk_ids = [row[0] for row in result]

                # 2) ベクトルストアから削除（あれば）
                if chunk_ids and self.vector_store:
                    try:
                        # PGVector.delete は ids を渡す
                        self.vector_store.delete(ids=chunk_ids, collection_name=coll_name)
                    except Exception as ve:
                        # 失敗しても SQL 側を削除し、メッセージに残す
                        warn = f"Vector store delete failed: {ve}"
                        print(warn)

                # 3) keyword検索テーブルから削除
                conn.execute(
                    text(
                        "DELETE FROM document_chunks "
                        "WHERE document_id = :doc_id AND collection_name = :coll_name"
                    ),
                    {"doc_id": doc_id, "coll_name": coll_name}
                )

                # 4) ベクトル埋め込みテーブルからも削除（メタデータ chunk_id ベース）
                # langchain_pg_embedding.cmetadata に chunk_id が入っている前提
                conn.execute(
                    text(
                        "DELETE FROM langchain_pg_embedding "
                        "WHERE collection_id = (SELECT uuid FROM langchain_pg_collection WHERE name = :coll_name) "
                        "AND (cmetadata ->> 'chunk_id') = ANY(:chunk_ids)"
                    ),
                    {"coll_name": coll_name, "chunk_ids": chunk_ids or []}
                )

                conn.commit()

            return True, f"Successfully deleted document: {doc_id} from collection: {coll_name}"

        except Exception as e:
            return False, f"Error deleting document: {str(e)}"
