#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Color helpers
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}🚀 Coronary Analyser - Web Deployment${NC}"
echo "========================================"

# Check .env file
if [ ! -f .env ]; then
    echo -e "${RED}❌ .env dosyası bulunamadı!${NC}"
    echo "   .env.example dosyasını kopyalayıp token'ı girin:"
    echo "   cp .env.example .env && nano .env"
    exit 1
fi

# Source .env
source .env

if [ -z "$TUNNEL_TOKEN" ]; then
    echo -e "${RED}❌ TUNNEL_TOKEN .env dosyasında tanımlı değil!${NC}"
    echo "   Cloudflare dashboard'dan tunnel token alın."
    exit 1
fi

echo -e "${YELLOW}📦 Docker imajları build ediliyor...${NC}"
VITE_API_BASE="" docker compose -f docker-compose.prod.yml build

echo -e "${YELLOW}🔄 Servisler başlatılıyor...${NC}"
docker compose -f docker-compose.prod.yml up -d

echo ""
echo -e "${GREEN}✅ Deployment tamamlandı!${NC}"
echo ""
echo "Servis durumları:"
docker compose -f docker-compose.prod.yml ps
echo ""
echo -e "Local:  ${GREEN}http://localhost:8080${NC}"
echo -e "Tunnel: ${GREEN}Cloudflare Dashboard'dan kontrol edin${NC}"
echo ""
echo "Logları görmek için:"
echo "  docker compose -f docker-compose.prod.yml logs -f"
echo ""
echo "Durdurmak için:"
echo "  docker compose -f docker-compose.prod.yml down"
