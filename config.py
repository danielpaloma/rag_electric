# Configuration file for model/provider constants

# General
DB_NAME = "rag_electric/vector_db"

# OpenAI
OPENAI_MODEL_ID = "gpt-4o-mini"  # or "gpt-3.5-turbo", etc.

# AWS Bedrock (Mistral Small)
#DOCS: https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-runtime_example_bedrock-runtime_InvokeModel_MistralAi_section.html
BEDROCK_MODEL_ID = "mistral.mistral-small-2402-v1:0"
BEDROCK_REGION = "us-east-1"
BEDROCK_MODEL_KWARGS = {"temperature": 0.7}

# Add more model/provider configs as needed 