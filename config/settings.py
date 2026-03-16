from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # API Keys
    anthropic_api_key: str = Field(..., env="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    pinecone_api_key: str = Field(..., env="PINECONE_API_KEY")
    cohere_api_key: str = Field(..., env="COHERE_API_KEY")

    # SEC EDGAR
    sec_user_agent_name: str = Field(..., env="SEC_USER_AGENT_NAME")
    sec_user_agent_email: str = Field(..., env="SEC_USER_AGENT_EMAIL")

    # Models
    claude_model: str = Field(default="claude-sonnet-4-6", env="CLAUDE_MODEL")
    embedding_model: str = Field(default="text-embedding-3-large", env="EMBEDDING_MODEL")
    embedding_dimensions: int = Field(default=3072, env="EMBEDDING_DIMENSIONS")

    # Pinecone
    pinecone_index_name: str = Field(default="investor-filings", env="PINECONE_INDEX_NAME")
    pinecone_region: str = Field(default="us-east-1", env="PINECONE_REGION")

    # Chunking
    chunk_overlap_pct: float = Field(default=0.10, env="CHUNK_OVERLAP_PCT")
    semantic_breakpoint_threshold: float = Field(default=95.0, env="SEMANTIC_BREAKPOINT_THRESHOLD")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
