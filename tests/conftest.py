from __future__ import annotations

import os

os.environ["ANTHROPIC_API_KEY"] = "test-key"
os.environ["PINECONE_API_KEY"] = "test-key"
os.environ["LANGCHAIN_API_KEY"] = "test-key"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["USE_PINECONE"] = "false"
os.environ["IS_TESTING"] = "true"
