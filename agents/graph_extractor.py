import json
import re
import hashlib
import time

class GraphExtractor:
    def __init__(self, llm):
        self.llm = llm

    def _make_id(self, name: str):
        return hashlib.md5(name.lower().encode()).hexdigest()

    def _extract_json_block(self, text: str):
        """
        Extract JSON object that contains "entities" and "relations".
        """
        import re

        pattern = re.compile(r'\{[\s\S]*?"entities"[\s\S]*?"relations"[\s\S]*?\}')
        match = pattern.search(text)

        if match:
            return match.group(0)

        return None




    def extract(self, text: str):
        start_time = time.time()

        print("\n" + "="*60)
        print("[GraphExtractor] 🚀 Extraction started")
        print(f"[GraphExtractor] Input text length: {len(text)} chars")

        MAX_CHARS = 400
        text_short = text[:MAX_CHARS].replace("\n", " ")
        print("[GraphExtractor] Text preview:")
        print(text_short + ("..." if len(text) > 300 else ""))
        prompt = f"""
You MUST return exactly ONE JSON object.

Do NOT include explanations.
Do NOT include multiple JSON objects.
Do NOT include markdown.

Return format:

{{
  "entities": [...],
  "relations": [...]
}}

Text:
{text_short}
"""




        # ---- LLM call ----
        try:
            llm_start = time.time()
            response = self.llm.complete(prompt).text.strip()
            print(f"[GraphExtractor] ✅ LLM call completed in {time.time() - llm_start:.2f}s")
        except Exception as e:
            print("[GraphExtractor] ❌ LLM call failed:", e)
            return {"entities": [], "relations": []}

        print("\n[GraphExtractor] Raw LLM output:")
        print("-" * 40)
        print(response)
        print("-" * 40)

        # ---- JSON extraction ----
        json_block = self._extract_json_block(response)
        if not json_block:
            print("[GraphExtractor] ❌ ERROR: No valid JSON block found")
            print("[GraphExtractor] Hint: model likely hallucinated or ignored format.")
            return {"entities": [], "relations": []}

        print("\n[GraphExtractor] Extracted JSON block:")
        print(json_block)

        # ---- JSON parsing ----
        try:
            data = json.loads(json_block)
        except Exception as e:
            print("[GraphExtractor] JSON parse failed, attempting repair...")

            try:
                # remove trailing garbage
                fixed = json_block.strip()
                fixed = fixed.replace("\n", " ")
                fixed = fixed.replace(",}", "}")
                fixed = fixed.replace(",]", "]")

                data = json.loads(fixed)
            except Exception as e2:
                print("[GraphExtractor] ❌ JSON repair failed:", e2)
                return {"entities": [], "relations": []}


        # ---- Entity processing ----
        entities = []
        for e in data.get("entities", []):
            name = str(e.get("name", "")).strip()
            if not name:
                print("[GraphExtractor] ⚠️ Skipping entity with empty name")
                continue

            entities.append({
                "id": self._make_id(name),
                "name": name,
                "type": e.get("type", "other")
            })

        # ---- Relation processing ----
        relations = []
        for r in data.get("relations", []):
            s = str(r.get("source", "")).strip()
            t = str(r.get("target", "")).strip()

            if not s or not t:
                print("[GraphExtractor] ⚠️ Skipping invalid relation:", r)
                continue

            relations.append({
                "source": self._make_id(s),
                "target": self._make_id(t),
                "source_name": s,
                "target_name": t,
                "relation": str(r.get("relation", "RELATED_TO")).upper()
            })

        # ---- Final stats ----
        duration = time.time() - start_time

        print("\n[GraphExtractor] 📊 Extraction summary")
        print(f"[GraphExtractor] Entities extracted : {len(entities)}")
        print(f"[GraphExtractor] Relations extracted: {len(relations)}")
        print(f"[GraphExtractor] Total time        : {duration:.2f}s")

        if not entities and not relations:
            print("[GraphExtractor] ⚠️ WARNING: Empty graph result")

        print("="*60)

        return {"entities": entities, "relations": relations}
