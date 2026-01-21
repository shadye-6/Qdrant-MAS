from neo4j import GraphDatabase

uri = "bolt://localhost:7687"
user = "neo4j"
password = "password"

try:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        result = session.run("RETURN 1")
        print("Neo4j connection OK:", result.single()[0])
except Exception as e:
    print("Neo4j connection FAILED:", e)
