#!/bin/bash

echo "🔄 Parando todos os containers..."
docker stop $(docker ps -aq) 2>/dev/null

echo "🧹 Removendo todos os containers..."
docker rm $(docker ps -aq) 2>/dev/null

echo "🧼 Removendo todas as imagens..."
docker rmi $(docker images -q) 2>/dev/null

echo "📁 Removendo todos os volumes..."
docker volume rm $(docker volume ls -q) 2>/dev/null

echo "🌐 Removendo todas as networks criadas manualmente..."
docker network rm $(docker network ls | grep -v 'bridge\|host\|none' | awk '{ if(NR>1) print $1 }') 2>/dev/null

echo "🧨 Limpando tudo com docker system prune..."
docker system prune -a --volumes -f

echo "✅ Docker limpo com sucesso!"
