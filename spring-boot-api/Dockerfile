# ── Stage 1: Build with Maven ──
FROM maven:3.9-eclipse-temurin-17 AS builder

WORKDIR /app

# Copy pom.xml first for dependency caching
COPY pom.xml .
RUN mvn dependency:go-offline -B

# Copy source code and build
COPY src ./src
RUN mvn package -DskipTests -B

# ── Stage 2: Lightweight runtime ──
FROM eclipse-temurin:17-jre

WORKDIR /app

# Copy the built JAR from the builder stage
COPY --from=builder /app/target/*.jar app.jar

# Spring Boot listens on 8080
EXPOSE 8080

# Allow gRPC host to be overridden via environment variable
# In Docker Compose, GRPC_SERVER_HOST will be set to "he-grpc-server"
ENV GRPC_SERVER_HOST=he-grpc-server
ENV GRPC_SERVER_PORT=50051

# Use shell form so environment variables are expanded at runtime
ENTRYPOINT java -jar app.jar \
  --grpc.server.host=${GRPC_SERVER_HOST} \
  --grpc.server.port=${GRPC_SERVER_PORT}
