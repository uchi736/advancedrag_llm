import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    # Database settings (populated in __post_init__)
    db_host: str = ""
    db_port: str = ""
    db_name: str = ""
    db_user: str = ""
    db_password: str = ""
    pgvector_connection_string: str = ""

    # OpenAI settings (populated in __post_init__)
    openai_api_key: Optional[str] = None

    # Azure OpenAI settings (populated in __post_init__)
    azure_openai_api_key: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None
    azure_openai_api_version: str = ""
    azure_openai_chat_deployment_name: Optional[str] = None
    azure_openai_embedding_deployment_name: Optional[str] = None

    # LLM Provider settings (populated in __post_init__)
    llm_provider: str = ""  # "azure" or "huggingface"

    # Hugging Face settings (populated in __post_init__)
    hf_model_id: str = ""
    hf_embedding_model_id: str = ""
    hf_device: str = ""  # "cpu", "cuda", "mps"
    hf_load_in_4bit: bool = False
    hf_load_in_8bit: bool = False
    hf_max_new_tokens: int = 0
    hf_temperature: float = 0.0
    hf_top_k: int = 0
    hf_top_p: float = 0.0

    # LLM settings (populated in __post_init__)
    llm_temperature: float = 0.0
    max_tokens: int = 0

    def __post_init__(self):
        """Load environment variables dynamically after load_dotenv() has been called"""
        # Database settings
        self.db_host = os.getenv("DB_HOST", "localhost")
        self.db_port = os.getenv("DB_PORT", "5432")
        self.db_name = os.getenv("DB_NAME", "postgres")
        self.db_user = os.getenv("DB_USER", "postgres")
        self.db_password = os.getenv("DB_PASSWORD", "your-password")

        # Build PGVector connection string
        try:
            import psycopg
            _PG_DIALECT = "psycopg"
        except ModuleNotFoundError:
            _PG_DIALECT = "psycopg2"

        self.pgvector_connection_string = (
            f"postgresql+{_PG_DIALECT}://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

        # OpenAI settings
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        # Azure OpenAI settings
        self.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        self.azure_openai_chat_deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
        self.azure_openai_embedding_deployment_name = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

        # LLM Provider settings
        self.llm_provider = os.getenv("LLM_PROVIDER", "azure")

        # Hugging Face settings
        self.hf_model_id = os.getenv("HF_MODEL_ID", "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3")
        self.hf_embedding_model_id = os.getenv("HF_EMBEDDING_MODEL_ID", "intfloat/multilingual-e5-large")
        self.hf_device = os.getenv("HF_DEVICE", "cuda")
        self.hf_load_in_4bit = os.getenv("HF_LOAD_IN_4BIT", "true").lower() == "true"
        self.hf_load_in_8bit = os.getenv("HF_LOAD_IN_8BIT", "false").lower() == "true"
        self.hf_max_new_tokens = int(os.getenv("HF_MAX_NEW_TOKENS", "2048"))
        self.hf_temperature = float(os.getenv("HF_TEMPERATURE", "0.0"))
        self.hf_top_k = int(os.getenv("HF_TOP_K", "50"))
        self.hf_top_p = float(os.getenv("HF_TOP_P", "0.9"))

        # LLM settings
        self.llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "16384"))

        # Azure Document Intelligence settings
        self.azure_di_endpoint = os.getenv("AZURE_DI_ENDPOINT")
        self.azure_di_api_key = os.getenv("AZURE_DI_API_KEY")
        self.azure_di_model = os.getenv("AZURE_DI_MODEL", "prebuilt-layout")
        self.save_markdown = os.getenv("SAVE_MARKDOWN", "false").lower() == "true"
        self.markdown_output_dir = os.getenv("MARKDOWN_OUTPUT_DIR", "output/markdown")

    # Azure Document Intelligence settings (populated in __post_init__)
    azure_di_endpoint: Optional[str] = None
    azure_di_api_key: Optional[str] = None
    azure_di_model: str = ""
    save_markdown: bool = False
    markdown_output_dir: str = ""

    # RAG and Search settings
    chunk_size: int = 1000
    chunk_overlap: int = 0  # No overlap to avoid duplicate content
    vector_search_k: int = 3
    keyword_search_k: int = 3
    final_k: int = 5
    collection_name: str = "documents"
    fts_language: str = "simple"  # Use 'simple' for language-agnostic search (supports Japanese, English, and mixed content)
    rrf_k_for_fusion: int = 60
    distance_strategy: str = "COSINE"
    vector_store_type: str = "pgvector"
    default_search_type: str = "hybrid"  # "vector", "keyword", or "hybrid"

    # Japanese search settings
    enable_japanese_search: bool = True
    japanese_min_token_length: int = 2

    # Jargon dictionary settings
    enable_jargon_extraction: bool = True
    enable_jargon_augmentation: bool = True
    jargon_table_name: str = "jargon_dictionary"
    max_jargon_terms_per_query: int = 5

    # Document processing settings
    enable_doc_summarization: bool = True
    enable_metadata_enrichment: bool = True
    confidence_threshold: float = 0.2

    # Term extraction settings (LLM-based)
    llm_extraction_chunk_size: int = 3000  # Text chunk size for LLM processing
    stage2_batch_size: int = 50  # Batch size for Stage 2 technical term filtering
