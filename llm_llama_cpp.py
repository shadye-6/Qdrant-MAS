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
    try:
        if hasattr(prompt, "template"):
            data = {}

            if hasattr(prompt, "kwargs") and prompt.kwargs:
                data.update(prompt.kwargs)

            if hasattr(prompt, "template_var_mappings") and prompt.template_var_mappings:
                data.update(prompt.template_var_mappings)

            data.setdefault("context_str", "")
            data.setdefault("query_str", "")

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

            # Context
            n_ctx=4096,

            # Threads / GPU
            n_threads=8,
            n_gpu_layers=999,   # set to 0 for CPU only

            # Determinism
            temperature=0.0,
            top_p=1.0,
            top_k=1,

            # Disable randomness
            repeat_penalty=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,

            # Prevent runaway
            max_tokens=256,

            # Formatting control
            stop=["</s>", "```", "\n\n\n"],

            verbose=False,
        )

        load_time = time.time() - start
        logger.info(f"Model loaded in {load_time:.2f}s")

        # ---- LlamaIndex metadata object ----
        class Meta:
            def __init__(self):
                self.context_window = 4096
                self.num_output = 256
                self.model_name = "llama-3-8b-instruct-gguf"

        self.metadata = Meta()

    # ----------------------------
    # Core generation
    # ----------------------------
    def complete(self, prompt):
        prompt_text = _render_prompt(prompt)

        logger.debug("LLM prompt length: %d chars", len(prompt_text))

        start = time.time()

        try:
            system_prompt = (
                "You are a strict information extraction engine. "
                "You ONLY output valid JSON according to the schema. "
                "You never answer questions. You never explain."
            )

            full_prompt = f"""<|begin_of_text|>
            <|system|>
            {system_prompt}
            <|user|>
            {prompt_text}
            <|assistant|>
            """

            out = self.llm(
                full_prompt,
                max_tokens=256,
                temperature=0.0,
                top_k=1,
                top_p=1.0,
                stop=["</s>", "<|eot_id|>", "<|end_of_text|>", "```"],
            )

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

        elapsed = time.time() - start
        logger.info(f"LLM generation completed in {elapsed:.2f}s")

        text = out["choices"][0]["text"].strip()

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
