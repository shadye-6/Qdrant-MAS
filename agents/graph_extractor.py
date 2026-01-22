import json
import hashlib
import time
import re


class GraphExtractor:
    def __init__(self, llm):
        self.llm = llm

    # ----------------------------
    # Utilities
    # ----------------------------
    def _make_id(self, name: str):
        return hashlib.md5(name.lower().encode()).hexdigest()

    def _extract_json_block(self, text: str):
        """
        Extract the first JSON object containing entities and relations.
        """
        pattern = re.compile(r'\{[\s\S]*?"entities"[\s\S]*?"relations"[\s\S]*?\}')
        match = pattern.search(text)
        return match.group(0) if match else None

    def _safe_parse_json(self, text: str):
        try:
            return json.loads(text)
        except Exception:
            block = self._extract_json_block(text)
            if not block:
                return None
            try:
                return json.loads(block)
            except Exception:
                # simple repair
                fixed = block.replace("\n", " ").replace(",}", "}").replace(",]", "]")
                try:
                    return json.loads(fixed)
                except Exception:
                    return None

    def _validate_schema(self, data: dict) -> bool:
        if not isinstance(data, dict):
            return False
        if "entities" not in data or "relations" not in data:
            return False
        if not isinstance(data["entities"], list) or not isinstance(data["relations"], list):
            return False

        for e in data["entities"]:
            if not isinstance(e, dict):
                return False
            if "name" not in e or "type" not in e:
                return False

        for r in data["relations"]:
            if not isinstance(r, dict):
                return False
            for k in ("source", "relation", "target"):
                if k not in r:
                    return False

        return True
    
    def _validate_consistency(self, data):
        entity_names = {e["name"] for e in data["entities"]}

        for r in data["relations"]:
            if r["source"] not in entity_names:
                return False
            if r["target"] not in entity_names:
                return False

        return True


    def _normalize(self, data):
        entities = []
        seen = set()

        for e in data["entities"]:
            name = e["name"].strip()
            if not name:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)

            entities.append({
                "id": self._make_id(name),
                "name": name,
                "type": e.get("type", "other").lower()
            })

        relations = []
        for r in data["relations"]:
            s = r["source"].strip()
            t = r["target"].strip()
            if not s or not t:
                continue

            relations.append({
                "source": self._make_id(s),
                "target": self._make_id(t),
                "source_name": s,
                "target_name": t,
                "relation": r["relation"].upper()
            })

        return entities, relations

    def _build_prompt(self, text_short: str) -> str:
        return f"""
You are an information extraction system.

ONLY extract facts explicitly stated in the text.
DO NOT infer.
DO NOT guess.
DO NOT add new entities.
DO NOT add background story.
DO NOT use world knowledge.

If something is not directly written, ignore it.

Output ONLY valid JSON.
No markdown.
No explanations.
No extra text.

SCHEMA:

{{
  "entities": [
    {{"name": "string", "type": "string"}}
  ],
  "relations": [
    {{"source": "string", "relation": "string", "target": "string"}}
  ]
}}

Relation rules:
- relation must be a verb phrase from the text
- use UPPER_SNAKE_CASE
- only use entities that appear in the text

If no relations exist, return empty arrays.

TEXT:
{text_short}
""".strip()

    # ----------------------------
    # Main extraction
    # ----------------------------
    def extract(self, text: str, max_retries: int = 1):
        start_time = time.time()

        print("\n" + "=" * 60)
        print("[GraphExtractor] 🚀 Extraction started")
        print(f"[GraphExtractor] Input text length: {len(text)} chars")

        text_short = text[:400].replace("\n", " ")

        for attempt in range(1, max_retries + 1):

            print(f"[GraphExtractor] Attempt {attempt}/{max_retries}")

            prompt = self._build_prompt(text_short)

            try:
                llm_start = time.time()
                response = self.llm.complete(prompt).text.strip()
                print(f"[GraphExtractor] ✅ LLM call completed in {time.time() - llm_start:.2f}s")
            except Exception as e:
                print("[GraphExtractor] ❌ LLM call failed:", e)
                continue

            print("\n[GraphExtractor] Raw LLM output:")
            print("-" * 40)
            print(response)
            print("-" * 40)

            data = self._safe_parse_json(response)

            if not data:
                print("[GraphExtractor] ❌ Invalid JSON")
                continue

            if not self._validate_schema(data):
                print("[GraphExtractor] ❌ Schema validation failed")
                continue

            # ---------- NEW: entity / relation consistency check ----------
            entity_names = {e["name"] for e in data.get("entities", [])}
            entities_list = data.get("entities", [])

            for r in data.get("relations", []):
                for role in ("source", "target"):
                    name = r.get(role)
                    if name not in entity_names:
                        print(f"[GraphExtractor] ⚠️ Auto-adding missing entity: {name}")
                        entities_list.append({
                            "name": name,
                            "type": "unknown"
                        })
                        entity_names.add(name)

            data["entities"] = entities_list

            # ------------------------------------------------------------

            entities, relations = self._normalize(data)

            duration = time.time() - start_time

            print("\n[GraphExtractor] 📊 Extraction summary")
            print(f"[GraphExtractor] Entities extracted : {len(entities)}")
            print(f"[GraphExtractor] Relations extracted: {len(relations)}")
            print(f"[GraphExtractor] Total time        : {duration:.2f}s")
            print("=" * 60)

            return {"entities": entities, "relations": relations}

        print("[GraphExtractor] ❌ Extraction failed after retries")
        print("=" * 60)
        return {"entities": [], "relations": []}
