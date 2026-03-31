# ── Stage 1: Build React app ──
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files first for dependency caching
COPY package.json package-lock.json* ./
RUN npm ci

# Copy source and build
COPY . .

# REACT_APP_API_BASE is empty so fetch calls use relative paths like "/api/..."
# Nginx will proxy /api/* to Spring Boot
ENV REACT_APP_API_BASE=/api
RUN npm run build

# ── Stage 2: Serve with Nginx ──
FROM nginx:alpine

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Copy our custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built React files
COPY --from=builder /app/build /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
