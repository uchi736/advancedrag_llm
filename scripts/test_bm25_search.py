"""
BM25キーワード検索 + ハイブリッド検索 + フィルタの包括的検証

Usage:
    python scripts/test_bm25_search.py
"""
import os
import sys
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from sqlalchemy import create_engine, text
from src.rag.config import Config
from src.rag.retriever import JapaneseHybridRetriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# Force UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

config = Config()
engine = create_engine(config.pgvector_connection_string)

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=config.azure_openai_endpoint,
    api_key=config.azure_openai_api_key,
    api_version=config.azure_openai_api_version,
    deployment=config.azure_openai_embedding_deployment_name
)

vector_store = PGVector(
    connection_string=config.pgvector_connection_string,
    embedding_function=embeddings,
    collection_name=config.collection_name
)

passed = 0
failed = 0


def test(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name} -- {detail}")


def make_retriever(**overrides):
    params = dict(
        vector_store=vector_store,
        connection_string=config.pgvector_connection_string,
        config_params=config,
        engine=engine,
    )
    params.update(overrides)
    return JapaneseHybridRetriever(**params)


# ================================================================
print("\n=== A. BM25 Keyword Search ===")
# ================================================================

retriever = make_retriever()

# A1: Japanese query
print("\n[A1] Japanese query")
kres = retriever._keyword_search("ガス軸受の特徴")
test("Japanese query returns results", len(kres) > 0, f"got {len(kres)}")
if kres:
    test("Results have content", len(kres[0][0].page_content) > 0)
    test("Results have score", kres[0][1] > 0)

# A2: English query
print("\n[A2] English query")
kres_en = retriever._keyword_search("carbon neutral")
test("English query does not crash", True)  # No exception = pass
print(f"  INFO: English results = {len(kres_en)}")

# A3: Empty collection
print("\n[A3] Empty collection")
config_empty = Config()
config_empty.collection_name = "nonexistent_collection_xyz"
r_empty = make_retriever(config_params=config_empty)
kres_empty = r_empty._keyword_search("test query")
test("Empty collection returns []", kres_empty == [], f"got {len(kres_empty)}")

# A4: Cache works
print("\n[A4] Cache behavior")
r_cache = make_retriever()
r_cache._keyword_search("ガス軸受")
first_retriever_id = id(r_cache.bm25_retriever)
r_cache._keyword_search("電動ターボ")
second_retriever_id = id(r_cache.bm25_retriever)
test("BM25 retriever cached (same object)", first_retriever_id == second_retriever_id)

# A5: Collection switch rebuilds
print("\n[A5] Collection switch")
r_switch = make_retriever()
r_switch._keyword_search("ガス軸受")
old_retriever_id = id(r_switch.bm25_retriever)
# Simulate switch to nonexistent collection
r_switch.config_params = Config()
r_switch.config_params.collection_name = "nonexistent_xyz"
kres_switch = r_switch._keyword_search("test")
test("Nonexistent collection returns empty", kres_switch == [])
# Switch back to original
r_switch.config_params.collection_name = config.collection_name
r_switch._keyword_search("ガス軸受")
test("Returns to original collection", r_switch.bm25_collection == config.collection_name)

# A6: Partial match returns results
print("\n[A6] Partial match (BM25 vs FTS)")
kres_partial = retriever._keyword_search("航空機の燃費向上")
test("Partial match returns results", len(kres_partial) > 0, f"got {len(kres_partial)}")

# ================================================================
print("\n=== B. Hybrid Search ===")
# ================================================================

# B1: RRF fusion
print("\n[B1] RRF fusion")
docs_hybrid = retriever._get_relevant_documents("ガス軸受の特徴")
test("Hybrid returns results", len(docs_hybrid) > 0, f"got {len(docs_hybrid)}")

# B2: Vector only
print("\n[B2] Vector only mode")
r_vec = make_retriever(search_type="vector")
docs_vec = r_vec._get_relevant_documents("ガス軸受の特徴")
test("Vector only returns results", len(docs_vec) > 0, f"got {len(docs_vec)}")

# B3: Keyword only
print("\n[B3] Keyword only mode")
r_kw = make_retriever(search_type="keyword")
docs_kw = r_kw._get_relevant_documents("ガス軸受の特徴")
test("Keyword only returns results", len(docs_kw) > 0, f"got {len(docs_kw)}")

# B4: Hybrid mode (default)
print("\n[B4] Hybrid mode (default)")
r_hyb = make_retriever()
test("Default search_type is hybrid", r_hyb.search_type == "hybrid")

# ================================================================
print("\n=== C. Vector Search Filter ===")
# ================================================================

# C1: jargon_term excluded
print("\n[C1] jargon_term exclusion")
vres = retriever._vector_search("電動ターボ機械")
types_in_results = [d.metadata.get('type') for d, _ in vres]
test("No jargon_term in vector results", 'jargon_term' not in types_in_results,
     f"types found: {set(types_in_results)}")

# C2: type=NULL not excluded
print("\n[C2] type=NULL handling")
with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT COUNT(*) FROM langchain_pg_embedding
        WHERE cmetadata->>'type' IS NULL
    """))
    null_count = result.scalar()
print(f"  INFO: Records with type=NULL: {null_count}")
# If NULL records exist, they should still appear in vector search
if null_count > 0:
    vres_all = vector_store.similarity_search_with_score("テスト", k=5)
    null_docs = [d for d, _ in vres_all if d.metadata.get('type') is None]
    test("NULL type docs included in raw search", len(null_docs) > 0)
else:
    test("No NULL type records (all updated to 'document')", True)

# C3: Stage3 post-filter
print("\n[C3] Stage3 post-filter simulation")
all_docs = vector_store.similarity_search("電動ターボ機械", k=20)
filtered = [d for d in all_docs if d.metadata.get('type') != 'jargon_term'][:5]
test("Post-filter returns document chunks", len(filtered) > 0, f"got {len(filtered)}")
if filtered:
    test("Filtered docs are not jargon_term",
         all(d.metadata.get('type') != 'jargon_term' for d in filtered))

# ================================================================
print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
print(f"{'='*50}")

sys.exit(1 if failed > 0 else 0)
