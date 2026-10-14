# Multi-stage Dockerfile for 100% Free 24/7 Hosting on Hugging Face Spaces
FROM node:20-alpine AS frontend-builder
WORKDIR /app/dashboard
COPY dashboard/package*.json ./
RUN npm ci
COPY dashboard/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app

# Install Nginx and Node.js (for Next.js runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Copy built frontend
COPY --from=frontend-builder /app/dashboard /app/dashboard

# Copy Nginx config and entrypoint
COPY nginx.conf /etc/nginx/nginx.conf
RUN chmod +x /app/entrypoint.sh

ENV PYTHONPATH=/app
ENV PORT=7860

# Expose Hugging Face default port
EXPOSE 7860

CMD ["/app/entrypoint.sh"]
