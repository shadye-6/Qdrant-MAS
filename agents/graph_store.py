from neo4j import GraphDatabase

class GraphStore:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        print("[GraphStore] Initializing connection to Neo4j...")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._ensure_schema()
        print("[GraphStore] Ready.")

    def close(self):
        print("[GraphStore] Closing connection.")
        self.driver.close()

    def _ensure_schema(self):
        print("[GraphStore] Ensuring schema constraints...")
        with self.driver.session() as session:
            session.run("""
            CREATE CONSTRAINT entity_id IF NOT EXISTS
            FOR (e:Entity) REQUIRE e.id IS UNIQUE
            """)
            session.run("""
            CREATE CONSTRAINT chunk_id IF NOT EXISTS
            FOR (c:Chunk) REQUIRE c.id IS UNIQUE
            """)
        print("[GraphStore] Schema OK.")

    def store(self, entities, relations, chunk_id):
        print("\n[GraphStore] store() called")
        print(f"[GraphStore] chunk_id = {chunk_id}")
        print(f"[GraphStore] entities ({len(entities)}): {entities}")
        print(f"[GraphStore] relations ({len(relations)}): {relations}")

        if not entities and not relations:
            print("[GraphStore] WARNING: Nothing to store (empty graph data)")
            return

        with self.driver.session() as session:

            # entities + chunk links
            for e in entities:
                print(f"[GraphStore] Storing entity: {e['name']}")
                session.run("""
                MERGE (n:Entity {id:$id})
                SET n.name=$name, n.type=$type
                MERGE (c:Chunk {id:$cid})
                MERGE (c)-[:MENTIONS]->(n)
                """, id=e["id"], name=e["name"], type=e["type"], cid=chunk_id)

            # relations
            for r in relations:
                print(f"[GraphStore] Storing relation: {r['source_name']} -[{r['relation']}]-> {r['target_name']}")
                session.run("""
                MERGE (a:Entity {id:$src})
                SET a.name=$src_name
                MERGE (b:Entity {id:$tgt})
                SET b.name=$tgt_name
                MERGE (a)-[:RELATION {type:$rel}]->(b)
                """,
                src=r["source"],
                tgt=r["target"],
                src_name=r["source_name"],
                tgt_name=r["target_name"],
                rel=r["relation"]
                )

        print("[GraphStore] Store operation finished.")

    def query(self, keywords, limit=15):
        print(f"\n[GraphStore] query() called with keywords: {keywords}")

        with self.driver.session() as session:
            res = session.run("""
            MATCH (a:Entity)-[r:RELATION]->(b:Entity)
            WHERE ANY(k IN $kw WHERE
                toLower(a.name) CONTAINS k OR
                toLower(b.name) CONTAINS k
            )
            RETURN a.name, r.type, b.name
            LIMIT $limit
            """, kw=keywords, limit=limit)

            results = [f"{r[0]} -[{r[1]}]-> {r[2]}" for r in res]

        print(f"[GraphStore] query() returned {len(results)} results")
        return results
