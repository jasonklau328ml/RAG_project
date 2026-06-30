from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from .config import DEFAULT_EMBED_MODEL_NAME, E5_QUERY_INSTRUCTION, E5_TEXT_INSTRUCTION


def create_embedding_model(model_name: str = DEFAULT_EMBED_MODEL_NAME) -> HuggingFaceEmbedding:
    """Create the embedding model used by both ingestion and retrieval.

    E5 models are trained with different prefixes for user queries and stored passages.
    Keeping this setup in one function prevents the insertion notebook and retrieval app
    from accidentally embedding text in incompatible ways.
    """
    if "e5" in model_name.lower():
        return HuggingFaceEmbedding(
            model_name=model_name,
            query_instruction=E5_QUERY_INSTRUCTION,
            text_instruction=E5_TEXT_INSTRUCTION,
            normalize=True,
        )

    return HuggingFaceEmbedding(model_name=model_name, normalize=True)