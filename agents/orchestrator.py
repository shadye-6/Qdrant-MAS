class OrchestratorAgent:
    def route(self, query: str):
        graph_triggers = [
            "relationship", "connect", "cause", "impact", "depend",
            "affect", "between", "related", "how does", "why does"
        ]

        q = query.lower()
        for t in graph_triggers:
            if t in q:
                return "graph"

        return "vector"
