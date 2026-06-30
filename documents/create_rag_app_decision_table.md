# create_rag_app Decision Table

This complements the flowchart by documenting conditional behavior and failure propagation.

| Decision Point | Condition | Action | Result |
|---|---|---|---|
| LLM provider branch | `llm_provider == "ollama"` | Build `Ollama(model=ollama_model)` | `resolved_llm_model = ollama_model` |
| LLM provider branch | `llm_provider == "huggingface"` | Resolve model id, load API key if needed, build `HuggingFaceChatLLM` | `resolved_llm_model = resolved Hugging Face model id` |
| LLM provider branch | Any other value | Raise `ValueError` in `create_llm` | `create_rag_app` fails |
| News directory check | `news_dir` exists | Continue configuration | Settings prepared |
| News directory check | `news_dir` missing | Raise `FileNotFoundError` in `configure_llama_index` | `create_rag_app` fails |
| Configuration outcome | `configure_llama_index` succeeds | Construct KB and session store | Return chatbot |
| Configuration outcome | `configure_llama_index` fails | Propagate exception | No chatbot returned |
| Constructor outcome | Any constructor fails (`ChromaKnowledgeBase`, `JsonChatSessionStore`, `RagNewsChatbot`) | Propagate exception | No chatbot returned |

## Practical review checklist
- Ensure `news_dir` is valid before call
- Ensure provider value is one of `ollama` or `huggingface`
- If using Hugging Face, ensure API key source is available
- Ensure `final_top_k` and `memory_token_limit` are sane for your workload
