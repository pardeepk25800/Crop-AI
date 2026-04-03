# ── Build Stage ──
FROM node:22-alpine as build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

# ── Serve Stage ──
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
# Update NGINX config to route requests to index.html for React Router
RUN echo 'server { listen 5173; location / { root /usr/share/nginx/html; index index.html; try_files $uri $uri/ /index.html; } }' > /etc/nginx/conf.d/default.conf
EXPOSE 5173
CMD ["nginx", "-g", "daemon off;"]
