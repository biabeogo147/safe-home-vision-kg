# Dockerfile for Neo4j deployment (optional - docker-compose.yml is preferred)
FROM neo4j:5.0-community

# Set environment variables
ENV NEO4J_AUTH=neo4j/hazarddetect123
ENV NEO4J_PLUGINS=["apoc"]
ENV NEO4J_dbms_security_procedures_unrestricted=apoc.*

# Expose ports
EXPOSE 7474 7687

# Copy any custom configuration or data
COPY ./neo4j.conf /var/lib/neo4j/conf/

# Entry point
CMD ["neo4j"]