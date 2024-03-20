from src.model.llm import LLM
from src.model.pt_llm import PromptTuningLLM
from src.model.graph_llm import GraphLLM


load_model = {
    'llm': LLM,
    'inference_llm': LLM,
    'pt_llm': PromptTuningLLM,
    'graph_llm': GraphLLM,
}

# Replace the following with the model paths
llama_model_path = {
    '7b': 'llama2/llama2_7b_hf',
    '13b': 'llama2/llama2_13b_hf',
    '7b_chat': 'meta-llama/Llama-2-7b-chat-hf',
    '13b_chat': 'llama2/llama2_13b_chat_hf',
    '7b-quant': 'TinyPixel/Llama-2-7B-bf16-sharded',
    'flan-t5-large': "google/flan-t5-large",
    'flan-t5-base' : 'google/flan-t5-base',
}
