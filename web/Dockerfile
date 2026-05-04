# --- Stage 1: build the SPA ---------------------------------------------------
FROM node:20-alpine AS build
WORKDIR /app

# Install deps first for better layer caching.
COPY package.json package-lock.json* ./
RUN npm install --no-audit --no-fund

# Build.
COPY . .
# Sub-path support: pass VITE_BASE_PATH=/medlit/ to deploy under a sub-path.
# VITE_API_BASE defaults to ${VITE_BASE_PATH}api so a single var is usually enough.
ARG VITE_BASE_PATH=/
ARG VITE_API_BASE
ENV VITE_BASE_PATH=$VITE_BASE_PATH
ENV VITE_API_BASE=${VITE_API_BASE:-${VITE_BASE_PATH}api}
RUN npm run build

# --- Stage 2: nginx serves static + reverse-proxies /api ----------------------
FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
HEALTHCHECK --interval=30s --timeout=3s CMD wget -qO- http://localhost/ || exit 1
