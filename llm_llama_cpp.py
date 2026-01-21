import time
import logging
from llama_cpp import Llama

# ----------------------------
# Logger setup
# ----------------------------
logger = logging.getLogger("LlamaCppLLM")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# ----------------------------
# Prompt rendering helper
# ----------------------------
def _render_prompt(prompt):
    """
    Safely render LlamaIndex PromptTemplate into a real string.
    Handles kwargs + template_var_mappings.
    """
    try:
        if hasattr(prompt, "template"):

            data = {}

            # normal kwargs (query_str, etc.)
            if hasattr(prompt, "kwargs") and prompt.kwargs:
                data.update(prompt.kwargs)

            # mapped variables (context_str, etc.)
            if hasattr(prompt, "template_var_mappings") and prompt.template_var_mappings:
                data.update(prompt.template_var_mappings)

            # ensure required keys exist
            if "context_str" not in data:
                data["context_str"] = ""

            if "query_str" not in data:
                data["query_str"] = ""

            return prompt.template.format(**data)

    except Exception as e:
        logger.warning(f"PromptTemplate rendering failed: {e}")

    return str(prompt)

# ----------------------------
# LLM Wrapper
# ----------------------------
class LlamaCppLLM:
    def __init__(self, model_path: str):
        logger.info("Loading llama.cpp model...")
        start = time.time()

        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=8,
            n_gpu_layers=40,   # set to 0 for CPU only
            verbose=False,
        )

        load_time = time.time() - start
        logger.info(f"Model loaded in {load_time:.2f}s")

        # ---- LlamaIndex metadata object ----
        class Meta:
            def __init__(self):
                self.context_window = 4096
                self.num_output = 256
                self.model_name = "mistral-7b-instruct-gguf"

        self.metadata = Meta()

    # ----------------------------
    # Core generation
    # ----------------------------
    def complete(self, prompt):
        prompt_text = _render_prompt(prompt)

        logger.debug("LLM prompt length: %d chars", len(prompt_text))

        start = time.time()

        try:
            out = self.llm(
                prompt_text,
                max_tokens=512,
                temperature=0.0,
                stop=["</s>", "```"],
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

        elapsed = time.time() - start
        logger.info(f"LLM generation completed in {elapsed:.2f}s")

        text = out["choices"][0]["text"].strip()

        # Remove common junk prefixes
        for bad in ("Output:", "Answer:", "Response:"):
            if text.startswith(bad):
                text = text[len(bad):].strip()

        class Resp:
            def __init__(self, t):
                self.text = t

        return Resp(text)

    # ----------------------------
    # LlamaIndex compatibility
    # ----------------------------
    def predict(self, prompt, **kwargs) -> str:
        prompt_text = _render_prompt(prompt)
        return self.complete(prompt_text).text
