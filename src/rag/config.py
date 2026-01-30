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
    llm_provider: Optional[str] = None  # "azure", "huggingface", "vllm", "openai", "gemini"

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

    # VLLM settings (OpenAI-compatible API via ChatOpenAI)
    vllm_endpoint: Optional[str] = None
    vllm_model: str = ""
    vllm_api_key: str = "EMPTY"
    vllm_temperature: float = 0.0
    vllm_max_tokens: int = 4096
    vllm_reasoning_effort: str = "medium"  # "low", "medium", "high"

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
        self.azure_openai_chat_mini_deployment_name = os.getenv("AZURE_OPENAI_CHAT_MINI_DEPLOYMENT_NAME")

        # LLM Provider settings (UIから渡された値を優先、空の場合のみ環境変数を使用)
        env_llm_provider = os.getenv("LLM_PROVIDER")
        if self.llm_provider not in (None, ""):
            # Value explicitly provided (e.g., from UI form) takes priority
            pass
        elif env_llm_provider:
            self.llm_provider = env_llm_provider
        else:
            self.llm_provider = "azure"

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

        # VLLM settings (OpenAI-compatible API)
        self.vllm_endpoint = os.getenv("VLLM_ENDPOINT")
        self.vllm_model = os.getenv("VLLM_MODEL", "")
        self.vllm_api_key = os.getenv("VLLM_API_KEY", "EMPTY")
        self.vllm_temperature = float(os.getenv("VLLM_TEMPERATURE", "0.0"))
        self.vllm_max_tokens = int(os.getenv("VLLM_MAX_TOKENS", "4096"))
        self.vllm_reasoning_effort = os.getenv("VLLM_REASONING_EFFORT", "medium")

        # LLM settings
        self.llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "16384"))

        # Azure Document Intelligence settings
        self.azure_di_endpoint = os.getenv("AZURE_DI_ENDPOINT")
        self.azure_di_api_key = os.getenv("AZURE_DI_API_KEY")
        self.azure_di_model = os.getenv("AZURE_DI_MODEL", "prebuilt-layout")
        self.save_markdown = os.getenv("SAVE_MARKDOWN", "false").lower() == "true"
        self.markdown_output_dir = os.getenv("MARKDOWN_OUTPUT_DIR", "output/markdown")

        # Domain classification settings
        self.stage4_domain_method = os.getenv("STAGE4_DOMAIN_METHOD", "llm")
        self.domain_min_terms_for_clustering = int(os.getenv("DOMAIN_MIN_TERMS_FOR_CLUSTERING", "10"))
        self.domain_hdbscan_min_cluster_size = int(os.getenv("DOMAIN_HDBSCAN_MIN_CLUSTER_SIZE", "2"))
        self.domain_cluster_selection_epsilon = float(os.getenv("DOMAIN_CLUSTER_SELECTION_EPSILON", "0.3"))
        self.enable_domain_cluster_naming = os.getenv("ENABLE_DOMAIN_CLUSTER_NAMING", "true").lower() == "true"

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

    # HyDE (Hypothetical Document Embeddings) settings
    enable_hyde: bool = False  # Generate hypothetical document for vector search

    # Document processing settings
    enable_doc_summarization: bool = True
    enable_metadata_enrichment: bool = True
    confidence_threshold: float = 0.2

    # Term extraction settings (LLM-based)
    llm_extraction_chunk_size: int = 3000  # Text chunk size for LLM processing
    stage2_batch_size: int = 50  # Batch size for Stage 2 technical term filtering
    max_concurrent_llm_calls: int = 20  # Max concurrent LLM calls (Windows select() limit workaround)
    enable_stage25_refinement: bool = True  # Stage 2.5 Self-Reflection ON/OFF
    max_refinement_iterations: int = 3  # Stage 2.5 最大反復回数
    reflection_batch_size_terms: int = 100  # Stage 2.5 反省バッチサイズ(terms)
    reflection_batch_size_candidates: int = 50  # Stage 2.5 反省バッチサイズ(candidates)

    # Domain classification settings (Stage 4)
    stage4_domain_method: str = "llm"  # "llm" | "hdbscan" | "hybrid"
    azure_openai_chat_mini_deployment_name: Optional[str] = None  # GPT-4o-mini for cluster naming
    domain_min_terms_for_clustering: int = 10  # Minimum terms for HDBSCAN clustering
    domain_hdbscan_min_cluster_size: int = 2  # HDBSCAN min_cluster_size
    domain_cluster_selection_epsilon: float = 0.3  # HDBSCAN cluster_selection_epsilon
    enable_domain_cluster_naming: bool = True  # LLM naming for clusters
