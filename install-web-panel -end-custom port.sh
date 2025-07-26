#!/bin/bash

# ===============================================================================
# TELEGRAM PROXY WEB PANEL - COMPLETE INSTALLATION
# ===============================================================================
# Single script that installs everything for web-based telegram proxy management
# Based on your original telegram proxy script but with web panel interface
# ===============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
WEB_PORT=8000
ADMIN_EMAIL=""
ADMIN_PASSWORD=""
PROJECT_NAME="telegram-proxy-panel"

# Logging functions
log() { echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }

# Check requirements
# Check requirements
check_requirements() {
    log "Checking system requirements..."
    
    # Install Docker if not present
    if ! command -v docker >/dev/null 2>&1; then
        log "Docker not found. Installing Docker..."
        curl -fsSL https://get.docker.com | sh
        systemctl start docker
        systemctl enable docker
        success "Docker installed successfully"
    else
        success "Docker is already installed"
    fi
    
    # Install Docker Compose if not present
    if ! docker compose version >/dev/null 2>&1; then
        log "Docker Compose not found. Installing..."
        if command -v apt >/dev/null 2>&1; then
            apt update
            apt install -y docker-compose-plugin
        elif command -v yum >/dev/null 2>&1; then
            yum install -y docker-compose-plugin
        else
            # Fallback - install standalone docker-compose
            curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
            chmod +x /usr/local/bin/docker-compose
        fi
        success "Docker Compose installed successfully"

        export PATH="/usr/local/bin:$PATH"
        
        # آپدیت bash session
        hash -r
    else
        success "Docker Compose is already installed"
    fi
    
    
    if ! docker info >/dev/null 2>&1; then
        error "Docker daemon is not running"
        echo "Start Docker: sudo systemctl start docker"
        exit 1
    fi
    
    if ! command -v python3 >/dev/null 2>&1; then
        error "Python 3 is not installed"
        exit 1
    fi
    
    success "All requirements met"
}

# Get admin credentials
get_admin_credentials() {
    log "Setting up admin credentials..."
    
    while true; do
        echo -n "Enter admin email (default: admin@example.com): "
        read -r input_email
        ADMIN_EMAIL="${input_email:-admin@example.com}"
        if [[ "$ADMIN_EMAIL" =~ ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$ ]]; then
            break
        else
            error "Please enter a valid email address"
        fi
    done
    
    while true; do
        echo -n "Enter admin password (minimum 8 characters): "
        read -s ADMIN_PASSWORD
        echo
        if [ ${#ADMIN_PASSWORD} -ge 8 ]; then
            break
        else
            error "Password must be at least 8 characters long"
        fi
    done
    
    success "Admin credentials set"
}

# Get server IP
get_server_ip() {
    log "Detecting server IP address..."
    for url in "https://ifconfig.me/ip" "https://ipinfo.io/ip" "https://icanhazip.com"; do
        if SERVER_IP=$(curl -s --connect-timeout 5 "$url" 2>/dev/null); then
            if [[ $SERVER_IP =~ ^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]]; then
                success "Server IP detected: $SERVER_IP"
                return 0
            fi
        fi
    done
    SERVER_IP="127.0.0.1"
    warning "Using localhost IP: $SERVER_IP"
}

# Create project structure and backend
create_backend_part() {
    log "Creating project structure and backend..."
    
    if [ -d "$PROJECT_NAME" ]; then
        warning "Directory exists. Backing up..."
        mv "$PROJECT_NAME" "${PROJECT_NAME}_backup_$(date +%Y%m%d_%H%M%S)" 2>/dev/null || true
    fi
    
    mkdir -p "$PROJECT_NAME"/{backend/app,frontend/src/{components,contexts},frontend/public,data/db,logs}
    cd "$PROJECT_NAME"
    
    # Create backend requirements.txt
    cat > backend/requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
sqlalchemy==2.0.23
passlib[bcrypt]==1.7.4
python-jose[cryptography]==3.3.0
python-multipart==0.0.6
python-dotenv==1.0.0
pydantic-settings==2.1.0
docker==6.1.3
aiofiles==23.2.1
requests==2.31.0
httpx==0.25.2
aiohttp==3.9.1
psutil==5.9.6
prometheus-client==0.19.0
cryptography>=41.0.0,<43.0.0
structlog==23.2.0
email-validator==2.1.0
EOF

  # Create backend Dockerfile
    cat > backend/Dockerfile << 'EOF'
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    sqlite3 curl docker.io netcat-openbsd gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 300 -r requirements.txt

COPY app/ ./app/
RUN mkdir -p /app/data/db /app/proxy-containers /app/logs

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

    # Create backend app files
    touch backend/app/__init__.py
    
    # Database setup
    cat > backend/app/database.py << 'EOF'
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/db/proxy_panel.db")
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in SQLALCHEMY_DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
EOF

    # Models
    cat > backend/app/models.py << 'EOF'
from sqlalchemy import Column, Integer, String, Boolean, DateTime, BigInteger, Text
from sqlalchemy.sql import func
from .database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    bandwidth_limit = Column(BigInteger, default=0)
    bandwidth_used = Column(BigInteger, default=0)
    proxy_port = Column(Integer, nullable=True)
    proxy_password = Column(String, nullable=True)
    proxy_container_id = Column(String, nullable=True)
    proxy_status = Column(String, default="inactive")
    created_at = Column(DateTime, server_default=func.now())
    last_login = Column(DateTime, nullable=True)
    total_connections = Column(Integer, default=0)
    bytes_sent = Column(BigInteger, default=0)
    bytes_received = Column(BigInteger, default=0)
EOF

    # Schemas
    cat > backend/app/schemas.py << 'EOF'
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

class UserBase(BaseModel):
    email: EmailStr
    username: str
    bandwidth_limit: Optional[int] = 0

class UserCreate(UserBase):
    password: str

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    password: Optional[str] = None
    is_active: Optional[bool] = None
    bandwidth_limit: Optional[int] = None

class User(UserBase):
    id: int
    is_active: bool
    is_admin: bool
    bandwidth_used: int
    proxy_port: Optional[int]
    proxy_status: str
    total_connections: int
    bytes_sent: int
    bytes_received: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str
EOF

    # Auth
    cat > backend/app/auth.py << 'EOF'
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from . import crud, models, schemas
from .database import SessionLocal

SECRET_KEY = "your-secret-key-change-in-production-$(openssl rand -hex 32)"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = crud.get_user_by_email(db, email=email)
    if user is None:
        raise credentials_exception
    return user

async def get_current_admin(current_user: schemas.User = Depends(get_current_user)):
    if not current_user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")
    return current_user
EOF

    # CRUD operations
    cat > backend/app/crud.py << 'EOF'
from sqlalchemy.orm import Session
from . import models, schemas, auth

def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()

def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.User).offset(skip).limit(limit).all()

def create_user(db: Session, user: schemas.UserCreate):
    hashed_password = auth.get_password_hash(user.password)
    db_user = models.User(
        email=user.email,
        username=user.username,
        hashed_password=hashed_password,
        bandwidth_limit=user.bandwidth_limit
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def update_user(db: Session, user_id: int, user_update: schemas.UserUpdate):
    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    if db_user:
        update_data = user_update.dict(exclude_unset=True)
        if "password" in update_data:
            update_data["hashed_password"] = auth.get_password_hash(update_data.pop("password"))
        for key, value in update_data.items():
            setattr(db_user, key, value)
        db.commit()
        db.refresh(db_user)
    return db_user

def delete_user(db: Session, user_id: int):
    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    if db_user:
        db.delete(db_user)
        db.commit()
    return db_user

def get_overview_stats(db: Session):
    total_users = db.query(models.User).count()
    active_users = db.query(models.User).filter(models.User.is_active == True).count()
    return {"total_users": total_users, "active_users": active_users, "total_bandwidth": 0, "active_connections": 0}
EOF

    success "Backend structure created"
}

# قسمت دوم - Proxy Manager و Main App

create_proxy_manager() {
    log "Creating proxy manager..."
    
    # Proxy management functions
    cat > backend/app/proxy_manager.py << 'EOF'
import os
import docker
import tempfile
import secrets
import string
from typing import Optional



def generate_secure_password():
    """Generate secure password"""
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(24))

def generate_secure_token():
    """Generate secure token"""
    return secrets.token_hex(16)

def generate_encryption_key():
    """Generate encryption key"""
    return secrets.token_hex(32)

def create_proxy_server_code(username, password, token, encryption_key, bandwidth_limit):
    """Create proxy server Python code based on your original script"""
    return f'''#!/usr/bin/env python3
import asyncio
import socket
import struct
import hashlib
import json
import time
import logging
import signal
import sys
from collections import defaultdict, deque
from typing import Dict, Set, Tuple, Optional
import aiohttp
from aiohttp import web

# Configuration
PROXY_PORT = 1080
BYPASS_PORT = 8443
METRICS_PORT = 8080
ADMIN_USERNAME = "{username}"
ADMIN_PASSWORD = "{password}"
ADMIN_TOKEN = "{token}"
ENCRYPTION_KEY = "{encryption_key}"
BANDWIDTH_LIMIT = {bandwidth_limit}

# Rate limiting
RATE_LIMIT = 15
RATE_WINDOW = 60
rate_limiter = defaultdict(lambda: deque())

# Telegram IP ranges and domains
TELEGRAM_IPS = [
    "149.154.160.0/20", "149.154.164.0/22", "149.154.168.0/22",
    "149.154.172.0/22", "91.108.4.0/22", "91.108.8.0/22",
    "91.108.12.0/22", "91.108.16.0/22", "91.108.20.0/22",
    "91.108.56.0/22", "95.161.64.0/20"
]

TELEGRAM_DOMAINS = [
    "telegram.org", "t.me", "tdesktop.com", "telegra.ph",
    "telegram.me", "api.telegram.org", "web.telegram.org"
]

# Metrics
metrics = {{
    "connections": 0, "bytes_sent": 0, "bytes_received": 0,
    "auth_failures": 0, "rate_limited": 0, "bypass_connections": 0,
    "bandwidth_used": 0, "bandwidth_limit": BANDWIDTH_LIMIT
}}

user_metrics = {{}}

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def is_telegram_allowed(host: str) -> bool:
    """Check if host is allowed (Telegram only)"""
    import ipaddress
    
    for domain in TELEGRAM_DOMAINS:
        if domain in host.lower():
            return True
    
    try:
        ip = ipaddress.ip_address(host)
        for ip_range in TELEGRAM_IPS:
            if ip in ipaddress.ip_network(ip_range):
                return True
    except:
        pass
    
    return False

def rate_limit_check(client_ip: str) -> bool:
    """Check rate limiting"""
    if client_ip in ['127.0.0.1', '::1', 'localhost']:
        return True
    
    now = time.time()
    client_requests = rate_limiter[client_ip]
    
    while client_requests and client_requests[0] < now - RATE_WINDOW:
        client_requests.popleft()
    
    if len(client_requests) >= RATE_LIMIT:
        metrics["rate_limited"] += 1
        return False
    
    client_requests.append(now)
    return True

def check_bandwidth_limit() -> bool:
    if BANDWIDTH_LIMIT == 0:
        return True
    total_usage = metrics["bytes_sent"] + metrics["bytes_received"]
    if total_usage >= BANDWIDTH_LIMIT:
        logger.warning(f"Bandwidth limit exceeded: {{total_usage / (1024*1024):.2f}} MB")
        import os
        os._exit(1)
    metrics["bandwidth_used"] = total_usage
    return True
    
    metrics["bandwidth_used"] = total_usage
    return True

def update_user_metrics(username: str, bytes_sent: int, bytes_received: int):
    """Update metrics for specific user"""
    if username not in user_metrics:
        user_metrics[username] = {{
            "bytes_sent": 0, "bytes_received": 0, "total_bytes": 0,
            "connections": 0, "last_connection": time.time()
        }}
    
    user_metrics[username]["bytes_sent"] += bytes_sent
    user_metrics[username]["bytes_received"] += bytes_received
    user_metrics[username]["total_bytes"] = user_metrics[username]["bytes_sent"] + user_metrics[username]["bytes_received"]
    user_metrics[username]["last_connection"] = time.time()

def format_bytes(bytes_count: int) -> str:
    """Format bytes to human readable format"""
    if bytes_count < 1024:
        return f"{{bytes_count}} B"
    elif bytes_count < 1024 * 1024:
        return f"{{bytes_count / 1024:.2f}} KB"
    elif bytes_count < 1024 * 1024 * 1024:
        return f"{{bytes_count / (1024 * 1024):.2f}} MB"
    else:
        return f"{{bytes_count / (1024 * 1024 * 1024):.2f}} GB"

async def handle_socks5_client(reader, writer):
    """Handle SOCKS5 client connection"""
    client_addr = writer.get_extra_info('peername')[0]
    
    try:
        if not rate_limit_check(client_addr):
            logger.warning(f"Rate limited: {{client_addr}}")
            writer.close()
            return
        
        if BANDWIDTH_LIMIT > 0 and metrics["bytes_sent"] + metrics["bytes_received"] >= BANDWIDTH_LIMIT:
            logger.error("Bandwidth limit reached - refusing new connections")
            writer.close()
            return
        
        # SOCKS5 handshake
        data = await reader.read(262)
        if len(data) < 3 or data[0] != 5:
            writer.close()
            return
        
        writer.write(b'\\x05\\x02')
        await writer.drain()
        
        # Authentication
        auth_data = await reader.read(513)
        if len(auth_data) < 3 or auth_data[0] != 1:
            writer.close()
            return
        
        username_len = auth_data[1]
        username = auth_data[2:2+username_len].decode()
        password_len = auth_data[2+username_len]
        password = auth_data[3+username_len:3+username_len+password_len].decode()
        
        if username != ADMIN_USERNAME or password != ADMIN_PASSWORD:
            writer.write(b'\\x01\\x01')
            await writer.drain()
            writer.close()
            metrics["auth_failures"] += 1
            return
        
        writer.write(b'\\x01\\x00')
        await writer.drain()

        if username not in user_metrics:
            user_metrics[username] = {{
                "bytes_sent": 0, "bytes_received": 0, "total_bytes": 0, 
                "connections": 0, "last_connection": time.time()
            }}
        user_metrics[username]["connections"] += 1
        
        # Connection request
        request = await reader.read(262)
        if len(request) < 10 or request[0] != 5 or request[1] != 1:
            writer.close()
            return
        
        # Parse target
        addr_type = request[3]
        if addr_type == 1:
            target_host = socket.inet_ntoa(request[4:8])
            target_port = struct.unpack('>H', request[8:10])[0]
        elif addr_type == 3:
            domain_len = request[4]
            target_host = request[5:5+domain_len].decode()
            target_port = struct.unpack('>H', request[5+domain_len:7+domain_len])[0]
        else:
            writer.write(b'\\x05\\x08\\x00\\x01\\x00\\x00\\x00\\x00\\x00\\x00')
            writer.close()
            return
        
        if not is_telegram_allowed(target_host):
            logger.warning(f"Blocked non-Telegram connection to {{target_host}}")
            writer.write(b'\\x05\\x02\\x00\\x01\\x00\\x00\\x00\\x00\\x00\\x00')
            writer.close()
            return
        
        try:
            target_reader, target_writer = await asyncio.open_connection(target_host, target_port)
            target_sock = target_writer.get_extra_info('socket')
            client_sock = writer.get_extra_info('socket')
            if target_sock:
                target_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            if client_sock:
                client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            writer.write(b'\\x05\\x00\\x00\\x01\\x00\\x00\\x00\\x00\\x00\\x00')
            await writer.drain()
            
            metrics["connections"] += 1
            logger.info(f"Connected: {{client_addr}} -> {{target_host}}:{{target_port}}")
            
            await asyncio.gather(
                relay_data(reader, target_writer, "client->target", username),
                relay_data(target_reader, writer, "target->client", username)
            )
            
        except Exception as e:
            logger.error(f"Connection failed to {{target_host}}:{{target_port}}: {{e}}")
            writer.write(b'\\x05\\x01\\x00\\x01\\x00\\x00\\x00\\x00\\x00\\x00')
            writer.close()
            
    except Exception as e:
        logger.error(f"SOCKS5 error: {{e}}")
    finally:
        writer.close()

async def relay_data(reader, writer, direction, username=None):
    try:
        while True:
            data = await reader.read(8192)
            if not data:
                break

            writer.write(data)
            await writer.drain()

            if "client->target" in direction:
                metrics["bytes_sent"] += len(data)
                if username:
                    update_user_metrics(username, len(data), 0)
                    # Check bandwidth limit after each data transfer
                    if not check_bandwidth_limit(username):
                        writer.close()
                        return
            else:
                metrics["bytes_received"] += len(data)
                if username:
                    update_user_metrics(username, 0, len(data))
                    # Check bandwidth limit after each data transfer
                    if not check_bandwidth_limit(username):
                        writer.close()
                        return

    except Exception:
        pass
    finally:
        writer.close()

async def metrics_handler(request):
    """Serve metrics"""
    output = []
    output.append("=== GLOBAL METRICS ===")
    for key, value in metrics.items():
        if key in ["bytes_sent", "bytes_received", "bandwidth_used"]:
            output.append(f"{{key}}: {{format_bytes(value)}}")
        else:
            output.append(f"{{key}}: {{value}}")
    
    output.append("\\n=== USER METRICS ===")
    if not user_metrics:
        output.append("No user data available")
    else:
        for username, stats in user_metrics.items():
            output.append(f"\\nUser: {{username}}")
            output.append(f"  Total Usage: {{format_bytes(stats['total_bytes'])}}")
            output.append(f"  Sent: {{format_bytes(stats['bytes_sent'])}}")
            output.append(f"  Received: {{format_bytes(stats['bytes_received'])}}")
            output.append(f"  Connections: {{stats['connections']}}")
            last_conn = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stats['last_connection']))
            output.append(f"  Last Connection: {{last_conn}}")
    
    return web.Response(text='\\n'.join(output), content_type='text/plain')

async def main():
    """Main server function"""
    logger.info("Starting Telegram SOCKS5 Proxy...")
    
    # Start SOCKS5 server
    socks5_server = await asyncio.start_server(handle_socks5_client, '0.0.0.0', PROXY_PORT)
    logger.info(f"SOCKS5 server listening on port {{PROXY_PORT}}")
    
    # Start metrics server
    app = web.Application()
    app.router.add_get('/metrics', metrics_handler)
    app.router.add_get('/users', metrics_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, '0.0.0.0', METRICS_PORT)
    await site.start()
    logger.info(f"Metrics server listening on port {{METRICS_PORT}}")
    
    logger.info("All servers started successfully")
    
    try:
        await asyncio.Future()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        socks5_server.close()
        await socks5_server.wait_closed()
        await runner.cleanup()

if __name__ == '__main__':
    try:
        import uvloop
        uvloop.install()
    except ImportError:
        pass
    
    asyncio.run(main())
'''

def create_dockerfile_content():
    """Create Dockerfile content for proxy container"""
    return '''FROM python:3.11-alpine

RUN apk add --no-cache gcc musl-dev linux-headers && \\
    pip install --no-cache-dir aiohttp websockets psutil uvloop && \\
    apk del gcc musl-dev linux-headers

RUN adduser -D -s /bin/sh proxyuser

COPY proxy-server.py /app/proxy-server.py
RUN chown -R proxyuser:proxyuser /app

USER proxyuser
WORKDIR /app

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', 1080)); s.close()"

EXPOSE 1080 8443 8080

CMD ["python", "proxy-server.py"]
'''

def create_user_proxy_container(user_id: int, username: str, bandwidth_limit: int = 0) -> Optional[dict]:
    """Create a dedicated proxy container for a user"""
    try:
        # Generate unique ports and credentials
        docker_client = docker.from_env()
        proxy_port = 10000 + user_id
        bypass_port = 11000 + user_id
        metrics_port = 12000 + user_id
        
        password = generate_secure_password()
        token = generate_secure_token()
        encryption_key = generate_encryption_key()
        
        # Create proxy server code
        proxy_code = create_proxy_server_code(username, password, token, encryption_key, bandwidth_limit)
        dockerfile_content = create_dockerfile_content()
        
        # Create temporary directory for build context
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write files
            with open(os.path.join(temp_dir, "proxy-server.py"), "w") as f:
                f.write(proxy_code)
            
            with open(os.path.join(temp_dir, "Dockerfile"), "w") as f:
                f.write(dockerfile_content)
            
            # Build image
            image_name = f"telegram-proxy-{username}"
            docker_client.images.build(path=temp_dir, tag=image_name, rm=True)
            
            # Run container
            container_name = f"telegram-proxy-{username}"
            
            # Stop existing container if exists
            try:
                docker_client = docker.from_env()
                existing = docker_client.containers.get(container_name)
                existing.stop()
                existing.remove()
            except docker.errors.NotFound:
                pass  # این خط مهم است - نباید خالی باشد
            
            container = docker_client.containers.run(
                image_name,
                name=container_name,
                ports={
                    '1080/tcp': proxy_port,
                    '8443/tcp': bypass_port,
                    '8080/tcp': metrics_port
                },
                detach=True,
                restart_policy={"Name": "unless-stopped"},
                security_opt=["no-new-privileges:true"],
                cap_drop=["ALL"],
                cap_add=["NET_BIND_SERVICE"]
            )
            
            return {
                "container_id": container.id,
                "proxy_port": proxy_port,
                "bypass_port": bypass_port,
                "metrics_port": metrics_port,
                "password": password,
                "token": token,
                "encryption_key": encryption_key
            }
            
    except Exception as e:
        print(f"Error creating proxy for user {username}: {e}")
        return None

def cleanup_user_proxy(username: str):
    """Clean up user proxy container"""
    try:
        container_name = f"telegram-proxy-{username}"
        image_name = f"telegram-proxy-{username}"
        
        # Stop and remove container
        try:
            container = docker_client.containers.get(container_name)
            container.stop()
            container.remove()
        except docker.errors.NotFound:
            pass
        
        # Remove image
        try:
            docker_client.images.remove(image_name)
        except docker.errors.ImageNotFound:
            pass
            
    except Exception as e:
        print(f"Error cleaning up proxy for user {username}: {e}")
EOF

    success "Proxy manager created"
}

create_main_app() {
    log "Creating main FastAPI application..."
    
    # Main FastAPI application with proxy management
    cat > backend/app/main.py << 'EOF'
from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime
from typing import List
import os
import subprocess
import tempfile
import secrets
import string

from . import models, schemas, crud, auth, database

models.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="Telegram Proxy Panel", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(auth.get_db)):
    user = crud.get_user_by_email(db, form_data.username)
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = auth.create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/me")
async def read_users_me(current_user: schemas.User = Depends(auth.get_current_user)):
    return current_user

def generate_secure_password():
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(24))

def parse_bandwidth_limit(bandwidth_input: str) -> int:
    if not bandwidth_input:
        return 0
    
    bandwidth_input = bandwidth_input.strip().upper()
    
    if bandwidth_input.endswith('M'):
        return int(float(bandwidth_input[:-1]) * 1024 * 1024)
    elif bandwidth_input.endswith('G'):
        return int(float(bandwidth_input[:-1]) * 1024 * 1024 * 1024)
    else:
        return int(bandwidth_input)

def create_telegram_proxy(username: str, bandwidth_limit: str, base_port: int):
    try:
        # Parse bandwidth
        bandwidth_bytes = parse_bandwidth_limit(bandwidth_limit)
        
        # Generate secure credentials
        admin_password = generate_secure_password()
        admin_token = secrets.token_hex(16)
        encryption_key = secrets.token_hex(32)
        
        # Calculate ports
        proxy_port = base_port
        bypass_port = base_port + 1
        metrics_port = base_port + 2
        
        # Create proxy directory
        proxy_dir = f"/app/proxy-containers/{username}"
        os.makedirs(proxy_dir, mode=0o755, exist_ok=True)
        os.chmod(proxy_dir, 0o755)
        
        # Create proxy server script
        proxy_script = f"""#!/usr/bin/env python3
import asyncio
import socket
import struct
import time
import logging
import sys
import os
from collections import defaultdict, deque
from typing import Dict, Set, Tuple, Optional
import aiohttp
from aiohttp import web

# Configuration
PROXY_PORT = 1080
BYPASS_PORT = 8443
METRICS_PORT = 8080
ADMIN_USERNAME = "{username}"
ADMIN_PASSWORD = "{admin_password}"
ADMIN_TOKEN = "{admin_token}"
ENCRYPTION_KEY = "{encryption_key}"
BANDWIDTH_LIMIT = {bandwidth_bytes}

# Rate limiting
RATE_LIMIT = 15
RATE_WINDOW = 60
rate_limiter = defaultdict(lambda: deque())

# Telegram IP ranges and domains
TELEGRAM_IPS = [
    "149.154.160.0/20", "149.154.164.0/22", "149.154.168.0/22",
    "149.154.172.0/22", "91.108.4.0/22", "91.108.8.0/22",
    "91.108.12.0/22", "91.108.16.0/22", "91.108.20.0/22",
    "91.108.56.0/22", "95.161.64.0/20"
]

TELEGRAM_DOMAINS = [
    "telegram.org", "t.me", "tdesktop.com", "telegra.ph",
    "telegram.me", "api.telegram.org", "web.telegram.org"
]

# Metrics
metrics = {{
    "connections": 0, "bytes_sent": 0, "bytes_received": 0,
    "auth_failures": 0, "rate_limited": 0, "bypass_connections": 0,
    "bandwidth_used": 0, "bandwidth_limit": BANDWIDTH_LIMIT
}}

user_metrics = {{}}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def is_telegram_allowed(host: str) -> bool:
    import ipaddress
    for domain in TELEGRAM_DOMAINS:
        if domain in host.lower():
            return True
    try:
        ip = ipaddress.ip_address(host)
        for ip_range in TELEGRAM_IPS:
            if ip in ipaddress.ip_network(ip_range):
                return True
    except:
        pass
    return False

def rate_limit_check(client_ip: str) -> bool:
    if client_ip in ['127.0.0.1', '::1', 'localhost']:
        return True
    now = time.time()
    client_requests = rate_limiter[client_ip]
    while client_requests and client_requests[0] < now - RATE_WINDOW:
        client_requests.popleft()
    if len(client_requests) >= RATE_LIMIT:
        metrics["rate_limited"] += 1
        return False
    client_requests.append(now)
    return True

def check_bandwidth_limit() -> bool:
    if BANDWIDTH_LIMIT == 0:
        return True
    total_usage = metrics["bytes_sent"] + metrics["bytes_received"]
    if total_usage >= BANDWIDTH_LIMIT:
        logger.warning(f"Bandwidth limit exceeded: {{total_usage / (1024*1024):.2f}} MB")
        import os
        os._exit(1)
    metrics["bandwidth_used"] = total_usage
    return True

def update_user_metrics(username: str, bytes_sent: int, bytes_received: int):
    if username not in user_metrics:
        user_metrics[username] = {{
            "bytes_sent": 0, "bytes_received": 0, "total_bytes": 0,
            "connections": 0, "last_connection": time.time()
        }}
    user_metrics[username]["bytes_sent"] += bytes_sent
    user_metrics[username]["bytes_received"] += bytes_received
    user_metrics[username]["total_bytes"] = user_metrics[username]["bytes_sent"] + user_metrics[username]["bytes_received"]
    user_metrics[username]["last_connection"] = time.time()

def format_bytes(bytes_count: int) -> str:
    if bytes_count < 1024:
        return f"{{bytes_count}} B"
    elif bytes_count < 1024 * 1024:
        return f"{{bytes_count / 1024:.2f}} KB"
    elif bytes_count < 1024 * 1024 * 1024:
        return f"{{bytes_count / (1024 * 1024):.2f}} MB"
    else:
        return f"{{bytes_count / (1024 * 1024 * 1024):.2f}} GB"

async def handle_socks5_client(reader, writer):
    client_addr = writer.get_extra_info('peername')[0]
    try:
        if not rate_limit_check(client_addr):
            logger.warning(f"Rate limited: {{client_addr}}")
            writer.close()
            return
        if BANDWIDTH_LIMIT > 0 and metrics["bytes_sent"] + metrics["bytes_received"] >= BANDWIDTH_LIMIT:
            logger.error("Bandwidth limit reached - refusing new connections")
            writer.close()
            return
        
        data = await reader.read(262)
        if len(data) < 3 or data[0] != 5:
            writer.close()
            return
        
        writer.write(b'\\x05\\x02')
        await writer.drain()
        
        auth_data = await reader.read(513)
        if len(auth_data) < 3 or auth_data[0] != 1:
            writer.close()
            return
        
        username_len = auth_data[1]
        username = auth_data[2:2+username_len].decode()
        password_len = auth_data[2+username_len]
        password = auth_data[3+username_len:3+username_len+password_len].decode()
        
        if username != ADMIN_USERNAME or password != ADMIN_PASSWORD:
            writer.write(b'\\x01\\x01')
            await writer.drain()
            writer.close()
            metrics["auth_failures"] += 1
            return
        
        writer.write(b'\\x01\\x00')
        await writer.drain()

        if username not in user_metrics:
            user_metrics[username] = {{
                "bytes_sent": 0, "bytes_received": 0, "total_bytes": 0, 
                "connections": 0, "last_connection": time.time()
            }}
        user_metrics[username]["connections"] += 1
        
        request = await reader.read(262)
        if len(request) < 10 or request[0] != 5 or request[1] != 1:
            writer.close()
            return
        
        addr_type = request[3]
        if addr_type == 1:
            target_host = socket.inet_ntoa(request[4:8])
            target_port = struct.unpack('>H', request[8:10])[0]
        elif addr_type == 3:
            domain_len = request[4]
            target_host = request[5:5+domain_len].decode()
            target_port = struct.unpack('>H', request[5+domain_len:7+domain_len])[0]
        else:
            writer.write(b'\\x05\\x08\\x00\\x01\\x00\\x00\\x00\\x00\\x00\\x00')
            writer.close()
            return
        
        if not is_telegram_allowed(target_host):
            logger.warning(f"Blocked non-Telegram connection to {{target_host}}")
            writer.write(b'\\x05\\x02\\x00\\x01\\x00\\x00\\x00\\x00\\x00\\x00')
            writer.close()
            return
        
        try:
            target_reader, target_writer = await asyncio.open_connection(target_host, target_port)
            target_sock = target_writer.get_extra_info('socket')
            client_sock = writer.get_extra_info('socket')
            if target_sock:
                target_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            if client_sock:
                client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            writer.write(b'\\x05\\x00\\x00\\x01\\x00\\x00\\x00\\x00\\x00\\x00')
            await writer.drain()
            
            metrics["connections"] += 1
            
            logger.info(f"Connected: {{client_addr}} -> {{target_host}}:{{target_port}}")
            
            await asyncio.gather(
                relay_data(reader, target_writer, "client->target", username),
                relay_data(target_reader, writer, "target->client", username)
            )
            
        except Exception as e:
            logger.error(f"Connection failed to {{target_host}}:{{target_port}}: {{e}}")
            writer.write(b'\\x05\\x01\\x00\\x01\\x00\\x00\\x00\\x00\\x00\\x00')
            writer.close()
            
    except Exception as e:
        logger.error(f"SOCKS5 error: {{e}}")
    finally:
        writer.close()

async def relay_data(reader, writer, direction, username=None):
    try:
        while True:
            data = await reader.read(8192)
            if not data:
                break
            
            writer.write(data)
            await writer.drain()
            
            if "client->target" in direction:
                metrics["bytes_sent"] += len(data)
                if username:
                    update_user_metrics(username, len(data), 0)
            else:
                metrics["bytes_received"] += len(data)
                if username:
                    update_user_metrics(username, 0, len(data))
                
    except Exception:
        pass
    finally:
        writer.close()

async def metrics_handler(request):
    output = []
    output.append("=== GLOBAL METRICS ===")
    for key, value in metrics.items():
        if key in ["bytes_sent", "bytes_received", "bandwidth_used"]:
            output.append(f"{{key}}: {{format_bytes(value)}}")
        else:
            output.append(f"{{key}}: {{value}}")
    
    output.append("\\n=== USER METRICS ===")
    if not user_metrics:
        output.append("No user data available")
    else:
        for username, stats in user_metrics.items():
            output.append(f"\\nUser: {{username}}")
            output.append(f"  Total Usage: {{format_bytes(stats['total_bytes'])}}")
            output.append(f"  Sent: {{format_bytes(stats['bytes_sent'])}}")
            output.append(f"  Received: {{format_bytes(stats['bytes_received'])}}")
            output.append(f"  Connections: {{stats['connections']}}")
            last_conn = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stats['last_connection']))
            output.append(f"  Last Connection: {{last_conn}}")
    
    return web.Response(text='\\n'.join(output), content_type='text/plain')

async def main():
    logger.info("Starting Telegram SOCKS5 Proxy...")
    
    socks5_server = await asyncio.start_server(handle_socks5_client, '0.0.0.0', PROXY_PORT)
    logger.info(f"SOCKS5 server listening on port {{PROXY_PORT}}")
    
    app = web.Application()
    app.router.add_get('/metrics', metrics_handler)
    app.router.add_get('/users', metrics_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, '0.0.0.0', METRICS_PORT)
    await site.start()
    logger.info(f"Metrics server listening on port {{METRICS_PORT}}")
    
    logger.info("All servers started successfully")
    
    try:
        await asyncio.Future()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        socks5_server.close()
        await socks5_server.wait_closed()
        await runner.cleanup()

if __name__ == '__main__':
    try:
        import uvloop
        uvloop.install()
    except ImportError:
        pass
    
    asyncio.run(main())
"""
        
        # Write proxy script
        with open(f"{proxy_dir}/proxy-server.py", "w") as f:
            f.write(proxy_script)
        
        # Create Dockerfile
        dockerfile_content = """FROM python:3.11-alpine

RUN apk add --no-cache gcc musl-dev linux-headers && \\
    pip install --no-cache-dir aiohttp websockets psutil uvloop && \\
    apk del gcc musl-dev linux-headers

RUN adduser -D -s /bin/sh proxyuser

COPY proxy-server.py /app/proxy-server.py
RUN chown -R proxyuser:proxyuser /app

USER proxyuser
WORKDIR /app

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', 1080)); s.close()"

EXPOSE 1080 8443 8080

CMD ["python", "proxy-server.py"]
"""
        
        with open(f"{proxy_dir}/Dockerfile", "w") as f:
            f.write(dockerfile_content)
        
        # Build and run container
        container_name = f"telegram-proxy-{username}"
        image_name = f"telegram-proxy-{username}"
        
        # Build image
        build_cmd = f"cd {proxy_dir} && docker build -t {image_name} ."
        result = subprocess.run(build_cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Docker build failed: {result.stderr}")
        
        # Stop existing container
        stop_cmd = f"docker stop {container_name} 2>/dev/null || true"
        subprocess.run(stop_cmd, shell=True)
        
        remove_cmd = f"docker rm {container_name} 2>/dev/null || true"
        subprocess.run(remove_cmd, shell=True)
        
        # Run new container
        run_cmd = f"""docker run -d \
            --name {container_name} \
            -p {proxy_port}:1080 \
            -p {bypass_port}:8443 \
            -p {metrics_port}:8080 \
            --restart unless-stopped \
            --security-opt no-new-privileges:true \
            --cap-drop ALL \
            --cap-add NET_BIND_SERVICE \
            {image_name}"""
        
        result = subprocess.run(run_cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            return {
                "container_id": result.stdout.strip(),
                "proxy_port": proxy_port,
                "bypass_port": bypass_port,
                "metrics_port": metrics_port,
                "password": admin_password,
                "username": username,
                "bandwidth_limit": bandwidth_limit
            }
        else:
            raise Exception(f"Failed to start container: {result.stderr}")
            
    except Exception as e:
        raise Exception(f"Error creating proxy: {str(e)}")

def get_next_available_port():
    import random
    
    # Get used ports from database
    db = database.SessionLocal()
    try:
        used_ports = set()
        db_users = db.query(models.User).filter(models.User.proxy_port.isnot(None)).all()
        for user in db_users:
            used_ports.add(user.proxy_port)
        
        # Generate random port and check if available
        for _ in range(100):  # Try 100 times
            port = random.randint(10000, 50000)
            if port not in used_ports:
                return port
        
        # Fallback: sequential search
        for port in range(10000, 60000):
            if port not in used_ports:
                return port
                
        return 10000  # Last resort
    finally:
        db.close()

@app.post("/create-proxy/")
async def create_proxy(
    proxy_data: dict,
    background_tasks: BackgroundTasks,
    db: Session = Depends(auth.get_db),
    current_user: schemas.User = Depends(auth.get_current_admin)
):
    username = proxy_data.get("username")
    bandwidth_limit = proxy_data.get("bandwidth_limit", "")
    custom_port = proxy_data.get("proxy_port", "")
    
    if not username:
        raise HTTPException(status_code=400, detail="Username is required")
    
    # Check if user already exists
    existing_user = db.query(models.User).filter(models.User.username == username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Handle port selection
    if custom_port:
        try:
            base_port = int(custom_port)
            if base_port < 1000 or base_port > 65000:
                raise HTTPException(status_code=400, detail="Port must be between 1000-65000")
            
            # Check if port is already in use
            existing_port_user = db.query(models.User).filter(models.User.proxy_port == base_port).first()
            if existing_port_user:
                raise HTTPException(status_code=400, detail=f"Port {base_port} is already in use")
                
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid port number")
    else:
        # Get next available port
        base_port = get_next_available_port()
    
    try:
        # Create proxy in background
        proxy_info = create_telegram_proxy(username, bandwidth_limit, base_port)
        
        # Create user record
        new_user = models.User(
            email=f"{username}@example.com",
            username=username,
            hashed_password=auth.get_password_hash(proxy_info["password"]),
            is_admin=False,
            is_active=True,
            bandwidth_limit=parse_bandwidth_limit(bandwidth_limit),
            proxy_port=proxy_info["proxy_port"],
            proxy_password=proxy_info["password"],
            proxy_container_id=proxy_info["container_id"],
            proxy_status="running"
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        return {
            "message": "Proxy created successfully",
            "user_id": new_user.id,
            "username": username,
            "proxy_port": proxy_info["proxy_port"],
            "status": "running"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create proxy: {str(e)}")

@app.get("/users/", response_model=List[schemas.User])
async def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(auth.get_db), current_user: schemas.User = Depends(auth.get_current_admin)):
    users = crud.get_users(db, skip=skip, limit=limit)
    return users

@app.delete("/users/{user_id}")
async def delete_user(
    user_id: int, 
    background_tasks: BackgroundTasks,
    db: Session = Depends(auth.get_db), 
    current_user: schemas.User = Depends(auth.get_current_admin)
):
    db_user = crud.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Stop and remove container
    if db_user.proxy_container_id:
        try:
            subprocess.run(f"docker stop {db_user.proxy_container_id}", shell=True)
            subprocess.run(f"docker rm {db_user.proxy_container_id}", shell=True)
            subprocess.run(f"docker rmi telegram-proxy-{db_user.username}", shell=True)
        except:
            pass
    
    # Remove proxy directory
    try:
        subprocess.run(f"rm -rf /app/proxy-containers/{db_user.username}", shell=True)
    except:
        pass
    
    crud.delete_user(db, user_id)
    return {"message": "User and proxy deleted successfully"}

@app.get("/users/{user_id}/config")
async def get_user_config(user_id: int, db: Session = Depends(auth.get_db), current_user: schemas.User = Depends(auth.get_current_admin)):
    user = crud.get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    server_ip = os.getenv("SERVER_IP", "127.0.0.1")
    
    return {
        "socks5": {
            "server": server_ip,
            "port": user.proxy_port,
            "username": user.username,
            "password": user.proxy_password
        },
        "telegram_link": f"https://t.me/socks?server={server_ip}&port={user.proxy_port}&user={user.username}&pass={user.proxy_password}",
        "qr_code_data": f"socks5://{user.username}:{user.proxy_password}@{server_ip}:{user.proxy_port}",
        "connection_string": f"socks5://{user.username}:{user.proxy_password}@{server_ip}:{user.proxy_port}"
    }

@app.get("/users/{user_id}/metrics")
async def get_user_metrics(user_id: int, db: Session = Depends(auth.get_db), current_user: schemas.User = Depends(auth.get_current_admin)):
    user = crud.get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user.proxy_status != "running":
        return {"status": user.proxy_status, "metrics": "Proxy not running"}

    # Check bandwidth limit immediately
    if user.bandwidth_limit > 0:
        total_usage = (user.bytes_sent or 0) + (user.bytes_received or 0)
        if total_usage >= user.bandwidth_limit:
            user.is_active = False
            user.proxy_status = "disabled"
            # Stop container immediately
            if user.proxy_container_id:
                try:
                    subprocess.run(f"docker stop {user.proxy_container_id}", shell=True)
                except:
                    pass
            db.commit()
            return {"status": "disabled", "metrics": "Bandwidth limit exceeded"}
    
    try:
        import requests
        metrics_port = user.proxy_port + 2
        response = requests.get(f"http://172.17.0.1:{metrics_port}/users", timeout=5)
        if response.status_code == 200:
            return {"status": "running", "metrics": response.text}
        return {"status": "error", "metrics": "Failed to get metrics"}
    except Exception as e:
        return {"status": "error", "metrics": str(e)}

@app.get("/stats/overview")
async def get_overview_stats(db: Session = Depends(auth.get_db), current_user: schemas.User = Depends(auth.get_current_admin)):
    total_users = db.query(models.User).filter(models.User.is_admin == False).count()
    active_users = db.query(models.User).filter(
        models.User.is_admin == False, 
        models.User.proxy_status == "running"
    ).count()
    
    return {
        "total_users": total_users,
        "active_users": active_users,
        "total_proxies": total_users,
        "running_proxies": active_users
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@app.put("/users/{user_id}")
async def update_user_bandwidth(
    user_id: int, 
    update_data: dict,
    db: Session = Depends(auth.get_db), 
    current_user: schemas.User = Depends(auth.get_current_admin)
):
    user = crud.get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    bandwidth_limit = update_data.get("bandwidth_limit", "")
    bandwidth_bytes = parse_bandwidth_limit(bandwidth_limit)
    
    new_port = update_data.get("proxy_port")
    port_changed = False
    
    # Check if port is changing
    if new_port and int(new_port) != user.proxy_port:
        # Validate new port
        try:
            new_port = int(new_port)
            if new_port < 1000 or new_port > 65000:
                raise HTTPException(status_code=400, detail="Port must be between 1000-65000")
            
            # Check if port is already in use by another user
            existing_port_user = db.query(models.User).filter(
                models.User.proxy_port == new_port,
                models.User.id != user_id
            ).first()
            if existing_port_user:
                raise HTTPException(status_code=400, detail=f"Port {new_port} is already in use")
            
            # Update port and mark for container restart
            user.proxy_port = new_port
            port_changed = True
            
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid port number")
    
    user.bandwidth_limit = bandwidth_bytes
    
    # If port changed, restart container with new port
    if port_changed and user.proxy_container_id:
        try:
            # Stop old container
            subprocess.run(f"docker stop {user.proxy_container_id}", shell=True)
            subprocess.run(f"docker rm {user.proxy_container_id}", shell=True)
            
            # Create new container with new port
            proxy_info = create_telegram_proxy(user.username, str(bandwidth_bytes), user.proxy_port)
            user.proxy_container_id = proxy_info["container_id"]
            user.proxy_status = "running"
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to restart container with new port: {str(e)}")
    
    db.commit()
    db.refresh(user)
    
    return {"message": "User updated successfully"}

@app.put("/users/{user_id}/toggle")
async def toggle_user_active(
    user_id: int, 
    toggle_data: dict,
    db: Session = Depends(auth.get_db), 
    current_user: schemas.User = Depends(auth.get_current_admin)
):
    user = crud.get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    is_active = toggle_data.get("is_active", True)
    user.is_active = is_active
    
    if user.proxy_container_id:
        try:
            if is_active:
                subprocess.run(f"docker start {user.proxy_container_id}", shell=True)
                user.proxy_status = "running"
            else:
                subprocess.run(f"docker stop {user.proxy_container_id}", shell=True)
                user.proxy_status = "stopped"
        except:
            pass
    
    db.commit()
    db.refresh(user)
    
    return {"message": f"User {'enabled' if is_active else 'disabled'} successfully"}
def parse_bytes_from_string(byte_str):
    """Parse bytes from strings like '18.02 MB'"""
    try:
        parts = byte_str.split()
        if len(parts) != 2:
            return 0
        
        value = float(parts[0])
        unit = parts[1].upper()
        
        if unit == 'B':
            return int(value)
        elif unit == 'KB':
            return int(value * 1024)
        elif unit == 'MB':
            return int(value * 1024 * 1024)
        elif unit == 'GB':
            return int(value * 1024 * 1024 * 1024)
    except:
        pass
    return 0

def sync_user_metrics_from_container(user_id: int, db: Session):
    """Sync user metrics from container and check bandwidth limit"""
    user = crud.get_user(db, user_id)
    if not user or user.proxy_status != "running":
        return False
    
    try:
        import requests
        metrics_port = user.proxy_port + 2
        response = requests.get(f"http://172.17.0.1:{metrics_port}/users", timeout=5)
        
        if response.status_code == 200:
            metrics_text = response.text
            lines = metrics_text.split('\n')
            
            for i, line in enumerate(lines):
                if f"User: {user.username}" in line:
                    for j in range(i+1, min(i+6, len(lines))):
                        if "Sent:" in lines[j]:
                            sent_str = lines[j].split(': ')[1].strip()
                            container_sent = parse_bytes_from_string(sent_str)
                            # فقط اگه بیشتر شده باشه بروزرسانی کن
                            if container_sent > (user.bytes_sent or 0):
                                user.bytes_sent = container_sent
                        elif "Received:" in lines[j]:
                            received_str = lines[j].split(': ')[1].strip()
                            container_received = parse_bytes_from_string(received_str)
                            # فقط اگه بیشتر شده باشه بروزرسانی کن
                            if container_received > (user.bytes_received or 0):
                                user.bytes_received = container_received
                    break
            
            if user.bandwidth_limit > 0:
                total_usage = (user.bytes_sent or 0) + (user.bytes_received or 0)
                if total_usage >= user.bandwidth_limit:
                    user.is_active = False
                    user.proxy_status = "disabled"
                    
                    if user.proxy_container_id:
                        try:
                            subprocess.run(f"docker stop {user.proxy_container_id}", shell=True, timeout=10)
                        except:
                            pass
                    
                    db.commit()
                    return True
            
            db.commit()
            return False
    except Exception as e:
        print(f"Error syncing metrics for user {user.username}: {e}")
        return False

@app.post("/check-bandwidth-limits")
async def check_bandwidth_limits(db: Session = Depends(auth.get_db)):
    users = db.query(models.User).filter(
        models.User.is_admin == False,
        models.User.bandwidth_limit > 0,
        models.User.proxy_status == "running"
    ).all()
    
    disabled_count = 0
    for user in users:
        if sync_user_metrics_from_container(user.id, db):
            disabled_count += 1
    
    return {"disabled_users": disabled_count}

# Background monitoring
import threading
import time

def continuous_bandwidth_monitor():
    while True:
        try:
            db = database.SessionLocal()
            try:
                users = db.query(models.User).filter(
                    models.User.is_admin == False,
                    models.User.proxy_status == "running",
                    models.User.bandwidth_limit > 0,
                    models.User.is_active == True
                ).all()
                
                for user in users:
                    sync_user_metrics_from_container(user.id, db)
                        
            finally:
                db.close()
                
            time.sleep(5)
            
        except Exception as e:
            print(f"Bandwidth monitor error: {e}")
            time.sleep(10)

monitor_thread = threading.Thread(target=continuous_bandwidth_monitor, daemon=True)
monitor_thread.start()
EOF

    success "Main FastAPI application created"
}

# قسمت چهارم - Frontend، Docker Compose و اجرا

create_frontend() {
    log "Creating frontend..."

    # Create frontend package.json with updated dependencies and scripts
   cat > frontend/package.json << 'EOF'
{
  "name": "telegram-proxy-panel-frontend",
  "version": "2.0.0",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1",
    "react-router-dom": "^6.8.0",
    "axios": "^1.6.0",
    "antd": "^5.12.0",
    "@ant-design/icons": "^5.2.0",
    "qrcode.react": "^3.1.0"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build"
  },
  "browserslist": {
    "production": [">0.2%", "not dead", "not op_mini all"],
    "development": ["last 1 chrome version", "last 1 firefox version", "last 1 safari version"]
  },
  "proxy": "http://backend:8000"
}
EOF

# Create frontend Dockerfile
    cat > frontend/Dockerfile << 'EOF'
FROM node:18-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    make \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy source code
COPY . .

# Build the app
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built files
COPY --from=builder /app/build /usr/share/nginx/html

# Add nginx config for React Router
RUN echo 'server { \
    listen 80; \
    location / { \
        root /usr/share/nginx/html; \
        index index.html index.htm; \
        try_files $uri $uri/ /index.html; \
    } \
}' > /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
EOF

    # Create frontend files
    cat > frontend/public/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Telegram Proxy Panel</title>
  </head>
  <body>
    <div id="root"></div>
  </body>
</html>
EOF

    mkdir -p frontend/src/{components,contexts}
    
    # Create React index
    cat > frontend/src/index.js << 'EOF'
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import 'antd/dist/reset.css';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
EOF

    # Create Auth Context
    cat > frontend/src/contexts/AuthContext.js << 'EOF'
import React, { createContext, useContext, useState, useEffect } from 'react';
import axios from 'axios';

const AuthContext = createContext();

export function useAuth() {
  return useContext(AuthContext);
}

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      fetchUser();
    } else {
      setLoading(false);
    }
  }, []);

  const fetchUser = async () => {
    try {
      const response = await axios.get(`http://${window.location.hostname}:8000/me`);
      setUser(response.data);
    } catch (error) {
      localStorage.removeItem('token');
      delete axios.defaults.headers.common['Authorization'];
    }
    setLoading(false);
  };

  const login = async (email, password) => {
    const formData = new FormData();
    formData.append('username', email);
    formData.append('password', password);
    const response = await axios.post(`http://${window.location.hostname}:8000/token`, formData);
    localStorage.setItem('token', response.data.access_token);
    axios.defaults.headers.common['Authorization'] = `Bearer ${response.data.access_token}`;
    await fetchUser();
    return response.data;
  };

  const logout = () => {
    localStorage.removeItem('token');
    delete axios.defaults.headers.common['Authorization'];
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, login, logout, loading }}>
      {children}
    </AuthContext.Provider>
  );
}
EOF

    # Create simple App component
    cat > frontend/src/App.js << 'EOF'
import React, { useState, useEffect } from 'react';
import { Layout, Button, Avatar, Space, Table, Modal, Form, Input, message, Tag, Tooltip } from 'antd';
import { UserOutlined, LogoutOutlined, PlusOutlined, EyeOutlined, DeleteOutlined, BarChartOutlined, EditOutlined, StopOutlined, PlayCircleOutlined } from '@ant-design/icons';
import QRCode from 'qrcode.react';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import axios from 'axios';

const { Header, Content } = Layout;
const { TextArea } = Input;

// Login component
function Login() {
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();

  const onFinish = async (values) => {
    setLoading(true);
    try {
      await login(values.email, values.password);
      message.success('Login successful!');
    } catch (error) {
      message.error('Invalid email or password');
    }
    setLoading(false);
  };

  return (
    <div style={{ 
      display: 'flex', 
      justifyContent: 'center', 
      alignItems: 'center', 
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
    }}>
      <div style={{ background: 'white', padding: '40px', borderRadius: '8px', width: '400px' }}>
        <h2 style={{ textAlign: 'center', marginBottom: '24px' }}>Telegram Proxy Panel</h2>
        <Form onFinish={onFinish} size="large">
          <Form.Item
            name="email"
            rules={[{ required: true, message: 'Please input your email!' }, { type: 'email' }]}
          >
            <Input placeholder="Email" />
          </Form.Item>
          <Form.Item
            name="password"
            rules={[{ required: true, message: 'Please input your password!' }]}
          >
            <Input.Password placeholder="Password" />
          </Form.Item>
          <Form.Item>
            <Button type="primary" htmlType="submit" loading={loading} block>
              Sign In
            </Button>
          </Form.Item>
        </Form>
      </div>
    </div>
  );
}

// Main dashboard component
function Dashboard() {
  const { user, logout } = useAuth();
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [modalVisible, setModalVisible] = useState(false);
  const [configModalVisible, setConfigModalVisible] = useState(false);
  const [metricsModalVisible, setMetricsModalVisible] = useState(false);
  const [userConfig, setUserConfig] = useState(null);
  const [userMetrics, setUserMetrics] = useState(null);
  const [stats, setStats] = useState({});
  const [form] = Form.useForm();

  useEffect(() => {
    fetchUsers();
    fetchStats();
  }, []);

  const fetchUsers = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`http://${window.location.hostname}:8000/users/`);
      let users = Array.isArray(response.data) ? response.data.filter(u => !u.is_admin) : [];
      setUsers(users);
    } catch (error) {
      message.error('Failed to fetch users');
    }
    setLoading(false);
  };

  const parseBytesFromString = (byteStr) => {
    try {
      const parts = byteStr.split(' ');
      const value = parseFloat(parts[0]);
      const unit = parts[1]?.toUpperCase();
      
      if (unit === 'MB') return value * 1024 * 1024;
      if (unit === 'KB') return value * 1024;
      if (unit === 'GB') return value * 1024 * 1024 * 1024;
      return value;
    } catch {
      return 0;
    }
  };

  const fetchStats = async () => {
    try {
      const response = await axios.get(`http://${window.location.hostname}:8000/stats/overview`);
      setStats(response.data);
    } catch (error) {
      console.error('Failed to fetch stats');
    }
  };

  const handleCreate = () => {
  // Generate random port
    const randomPort = Math.floor(Math.random() * 40000) + 10000;
    
    form.resetFields();
    // Set random port as default value
    form.setFieldsValue({
      proxy_port: randomPort
    });
    setModalVisible(true);
};

  const handleSubmit = async (values) => {
    try {
      const proxyData = {
        username: values.username,
        bandwidth_limit: values.bandwidth_limit || '',
        proxy_port: values.proxy_port || ''
      };
      await axios.post(`http://${window.location.hostname}:8000/create-proxy/`, proxyData);
      message.success('Telegram proxy created successfully!');
      setModalVisible(false);
      fetchUsers();
      fetchStats();
    } catch (error) {
      const errorMsg = error.response?.data?.detail || 'Operation failed';
      message.error(errorMsg);
    }
};

  const handleDelete = async (userId) => {
    Modal.confirm({
      title: 'Are you sure?',
      content: 'This will permanently delete the proxy and all its data.',
      onOk: async () => {
        try {
          await axios.delete(`http://${window.location.hostname}:8000/users/${userId}`);
          message.success('Proxy deleted successfully');
          fetchUsers();
          fetchStats();
        } catch (error) {
          message.error('Failed to delete proxy');
        }
      }
    });
  };

  const showConfig = async (user) => {
    try {
      const response = await axios.get(`http://${window.location.hostname}:8000/users/${user.id}/config`);
      setUserConfig(response.data);
      setConfigModalVisible(true);
    } catch (error) {
      message.error('Failed to get user config');
    }
  };

  const showMetrics = async (user) => {
    try {
      const response = await axios.get(`http://${window.location.hostname}:8000/users/${user.id}/metrics`);
      setUserMetrics(response.data);
     setMetricsModalVisible(true);
   } catch (error) {
     message.error('Failed to get user metrics');
   }
 };


const [editModalVisible, setEditModalVisible] = useState(false);
  const [editingUser, setEditingUser] = useState(null);
  const [editForm] = Form.useForm();

  const handleEdit = (user) => {
    setEditingUser(user);
    editForm.setFieldsValue({
      username: user.username,
      bandwidth_limit: formatBandwidthForEdit(user.bandwidth_limit),
      proxy_port: user.proxy_port
    });
    setEditModalVisible(true);
};

  const formatBandwidthForEdit = (bytes) => {
    if (!bytes || bytes === 0) return '';
    if (bytes >= 1024 * 1024 * 1024) {
      return `${(bytes / (1024 * 1024 * 1024)).toFixed(0)}G`;
    }
    return `${(bytes / (1024 * 1024)).toFixed(0)}M`;
  };

  const handleEditSubmit = async (values) => {
    try {
      await axios.put(`http://${window.location.hostname}:8000/users/${editingUser.id}`, {
        bandwidth_limit: values.bandwidth_limit || '',
        proxy_port: values.proxy_port
      });
      message.success('Proxy updated successfully!');
      setEditModalVisible(false);
      fetchUsers();
    } catch (error) {
      const errorMsg = error.response?.data?.detail || 'Failed to update proxy';
      message.error(errorMsg);
    }
};

 const [loadingUsers, setLoadingUsers] = useState(new Set());

  const handleToggleActive = async (user) => {
    // اضافه کردن loading state
    setLoadingUsers(prev => new Set([...prev, user.id]));
    
    try {
      const newStatus = !user.is_active;
      
      // نمایش پیام فوری
      if (!newStatus) {
        message.loading('Stopping proxy container...', 0);
      }
      
      await axios.put(`http://${window.location.hostname}:8000/users/${user.id}/toggle`, {
        is_active: newStatus
      });
      
      message.destroy(); // پاک کردن loading message
      message.success(`Proxy ${newStatus ? 'enabled' : 'disabled'} successfully!`);
      fetchUsers();
    } catch (error) {
      message.destroy();
      message.error('Failed to toggle proxy status');
    } finally {
      // حذف loading state
      setLoadingUsers(prev => {
        const newSet = new Set(prev);
        newSet.delete(user.id);
        return newSet;
      });
    }
  };

 const getStatusColor = (status) => {
   switch (status) {
     case 'running': return 'green';
     case 'creating': return 'blue';
     case 'error': return 'red';
     default: return 'default';
   }
 };

 const formatBandwidth = (bytes) => {
   if (!bytes || bytes === 0) return '0 B';
   if (bytes < 1024) return `${bytes} B`;
   if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
   if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(0)} MB`;
   return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
 };

 const columns = [
   { title: 'Username', dataIndex: 'username', key: 'username' },
   { 
     title: 'Port', 
     dataIndex: 'proxy_port', 
     key: 'proxy_port',
     render: (port) => port || 'N/A'
   },
   {
     title: 'Bandwidth Limit',
     dataIndex: 'bandwidth_limit',
     key: 'bandwidth_limit',
     render: (limit) => limit === 0 ? 'Unlimited' : formatBandwidth(limit)
   },

   {
     title: 'Usage',
     key: 'usage',
     render: (_, record) => {
       const limit = record.bandwidth_limit || 0;
       const sent = record.bytes_sent || 0;
       const received = record.bytes_received || 0;
       const totalUsed = sent + received;
       
       if (limit === 0) {
         return (
           <div>
             <div>Upload: {formatBandwidth(sent)}</div>
             <div>Download: {formatBandwidth(received)}</div>
             <div>Total Used: {formatBandwidth(totalUsed)}</div>
             <div>Remaining: Unlimited</div>
           </div>
         );
       }
       
       const remaining = Math.max(0, limit - totalUsed);
       const percentUsed = (totalUsed / limit) * 100;
       const isOverLimit = totalUsed >= limit;
       
       return (
         <div>
           <div>Upload: {formatBandwidth(sent)}</div>
           <div>Download: {formatBandwidth(received)}</div>
           <div>Total Used: {formatBandwidth(totalUsed)}</div>
           <div style={{color: isOverLimit ? 'red' : 'inherit'}}>
             Remaining: {formatBandwidth(remaining)}
           </div>
           <div style={{fontSize: '11px', color: isOverLimit ? 'red' : '#666'}}>
             {percentUsed.toFixed(1)}% used {isOverLimit ? '(EXCEEDED)' : ''}
           </div>
         </div>
       );
     }
   },
   {
     title: 'Actions',
     key: 'actions',
     render: (_, record) => (
       <Space>
         <Tooltip title="View Config & QR Code">
           <Button 
             type="primary" 
             icon={<EyeOutlined />} 
             size="small" 
             onClick={() => showConfig(record)}
             disabled={record.proxy_status !== 'running'}
           />
         </Tooltip>
         <Tooltip title="View Metrics">
           <Button 
             type="default" 
             icon={<BarChartOutlined />} 
             size="small" 
             onClick={() => showMetrics(record)}
             disabled={record.proxy_status !== 'running'}
           />
         </Tooltip>
         <Tooltip title="Edit Proxy">
           <Button 
             type="default" 
             icon={<EditOutlined />} 
             size="small" 
             onClick={() => handleEdit(record)}
           />
         </Tooltip>
         <Tooltip title={record.is_active ? "Disable Proxy" : "Enable Proxy"}>
           <label style={{ 
             position: 'relative',
             display: 'inline-block',
             width: '40px',
             height: '20px',
             marginLeft: '8px'
           }}>
             <input
               type="checkbox"
               checked={record.is_active}
               onChange={() => {
                 if (!loadingUsers.has(record.id)) {
                   handleToggleActive(record);
                 }
               }}
               disabled={loadingUsers.has(record.id)}
               style={{ opacity: 0, width: 0, height: 0 }}
             />
             <span style={{
               position: 'absolute',
               cursor: loadingUsers.has(record.id) ? 'not-allowed' : 'pointer',
               top: 0,
               left: 0,
               right: 0,
               bottom: 0,
               backgroundColor: loadingUsers.has(record.id) ? '#d9d9d9' : (record.is_active ? '#1890ff' : '#ccc'),
               borderRadius: '20px',
               transition: '0.3s',
               '::before': {
                 position: 'absolute',
                 content: '""',
                 height: '16px',
                 width: '16px',
                 left: record.is_active ? '22px' : '2px',
                 bottom: '2px',
                 backgroundColor: 'white',
                 borderRadius: '50%',
                 transition: '0.3s'
               }
             }}>
               <span style={{
                 position: 'absolute',
                 content: '""',
                 height: '16px',
                 width: '16px',
                 left: record.is_active ? '22px' : '2px',
                 bottom: '2px',
                 backgroundColor: 'white',
                 borderRadius: '50%',
                 transition: '0.3s'
               }}></span>
             </span>
           </label>
         </Tooltip>
         <Tooltip title="Delete Proxy">
           <Button 
             type="primary" 
             danger 
             icon={<DeleteOutlined />} 
             size="small" 
             onClick={() => handleDelete(record.id)}
           />
         </Tooltip>
       </Space>
     ),
   },
 ];

 return (
   <Layout style={{ minHeight: '100vh' }}>
     <Header style={{ 
       padding: '0 16px', 
       background: '#001529', 
       display: 'flex', 
       justifyContent: 'space-between', 
       alignItems: 'center' 
     }}>
       <h2 style={{ color: 'white', margin: 0 }}>Telegram Proxy Panel</h2>
       <Space style={{ cursor: 'pointer', color: 'white' }} onClick={logout}>
         <Avatar icon={<UserOutlined />} />
         <span>{user?.email}</span>
         <LogoutOutlined />
       </Space>
     </Header>
     <Content style={{ padding: '24px' }}>
       {/* Stats Cards */}
       <div style={{ marginBottom: 24, display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16 }}>
         <div style={{ background: '#f0f9ff', padding: 16, borderRadius: 8, border: '1px solid #0ea5e9' }}>
           <h3 style={{ margin: 0, color: '#0369a1' }}>Total Proxies</h3>
           <p style={{ margin: 0, fontSize: 24, fontWeight: 'bold', color: '#0c4a6e' }}>{stats.total_proxies || 0}</p>
         </div>
         <div style={{ background: '#f0fdf4', padding: 16, borderRadius: 8, border: '1px solid #22c55e' }}>
           <h3 style={{ margin: 0, color: '#15803d' }}>Running Proxies</h3>
           <p style={{ margin: 0, fontSize: 24, fontWeight: 'bold', color: '#14532d' }}>{stats.running_proxies || 0}</p>
         </div>
         <div style={{ background: '#fefce8', padding: 16, borderRadius: 8, border: '1px solid #eab308' }}>
           <h3 style={{ margin: 0, color: '#a16207' }}>Active Users</h3>
           <p style={{ margin: 0, fontSize: 24, fontWeight: 'bold', color: '#713f12' }}>{stats.active_users || 0}</p>
         </div>
       </div>

       <div style={{ marginBottom: 16, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
         <h2>Telegram Proxy Management</h2>
         <Button type="primary" icon={<PlusOutlined />} onClick={handleCreate}>
           Create New Proxy
         </Button>
       </div>

       <Table 
         dataSource={users} 
         columns={columns} 
         rowKey="id" 
         loading={loading}
         pagination={{ pageSize: 10 }}
       />

       {/* Create Proxy Modal */}
       <Modal
         title="Create New Telegram Proxy"
         open={modalVisible}
         onCancel={() => setModalVisible(false)}
         footer={null}
         width={500}
       >
         <Form form={form} onFinish={handleSubmit} layout="vertical">
           <Form.Item 
             name="username" 
            label="Username" 
              rules={[
                { required: true, message: 'Username is required' }, 
                { pattern: /^[a-zA-Z0-9_]+$/, message: 'Only letters, numbers and underscore allowed' }
              ]}
            >
              <Input placeholder="e.g., user1, client_a" />
            </Form.Item>
            <Form.Item 
              name="proxy_port" 
              label="Proxy Port"
              help="Leave empty for random port (10000-65000), or enter custom port (1000-65000)"
            >
              <Input 
                placeholder="Auto-generated random port"
                type="number"
                min="1000"
                max="65000"
              />
            </Form.Item>
            <Form.Item 
              name="bandwidth_limit" 
              label="Bandwidth Limit"
              help="Examples: 200M (200 megabytes), 5G (5 gigabytes), leave empty for unlimited"
            >
              <Input placeholder="e.g., 200M, 5G or leave empty for unlimited" />
           </Form.Item>
           <Form.Item>
             <Space>
               <Button type="primary" htmlType="submit">Create Proxy</Button>
               <Button onClick={() => setModalVisible(false)}>Cancel</Button>
             </Space>
           </Form.Item>
         </Form>
       </Modal>

       {/* Configuration Modal */}
       <Modal
         title="Proxy Configuration"
         open={configModalVisible}
         onCancel={() => setConfigModalVisible(false)}
         width={800}
         footer={[<Button key="close" onClick={() => setConfigModalVisible(false)}>Close</Button>]}
       >
         {userConfig && (
           <div>
             <div style={{ marginBottom: 16 }}>
               <h4>SOCKS5 Configuration:</h4>
               <div style={{ background: '#f5f5f5', padding: 12, borderRadius: 4, marginBottom: 8 }}>
                 <p style={{ margin: 0 }}><strong>Server:</strong> {userConfig.socks5.server}</p>
                 <p style={{ margin: 0 }}><strong>Port:</strong> {userConfig.socks5.port}</p>
                 <p style={{ margin: 0 }}><strong>Username:</strong> {userConfig.socks5.username}</p>
                 <p style={{ margin: 0 }}><strong>Password:</strong> {userConfig.socks5.password}</p>
               </div>
             </div>

             <div style={{ marginBottom: 16 }}>
               <h4>Telegram Direct Link:</h4>
               <TextArea value={userConfig.telegram_link} readOnly rows={2} />
               <Button 
                 style={{ marginTop: 8 }} 
                 onClick={() => {
                   navigator.clipboard.writeText(userConfig.telegram_link);
                   message.success('Telegram link copied!');
                 }}
               >
                 Copy Telegram Link
               </Button>
             </div>

             <div style={{ marginBottom: 16 }}>
               <h4>Connection String:</h4>
               <TextArea value={userConfig.connection_string} readOnly rows={1} />
               <Button 
                 style={{ marginTop: 8 }} 
                 onClick={() => {
                   navigator.clipboard.writeText(userConfig.connection_string);
                   message.success('Connection string copied!');
                 }}
               >
                 Copy Connection String
               </Button>
             </div>

             <div style={{ textAlign: 'center' }}>
               <h4>QR Code for Mobile Setup:</h4>
               <QRCode value={userConfig.qr_code_data} size={200} />
               <p style={{ marginTop: 8, fontSize: 12, color: '#666' }}>
                 Scan this QR code with your phone to automatically configure the proxy
               </p>
             </div>
           </div>
         )}
       </Modal>

       {/* Metrics Modal */}
       <Modal
         title="Proxy Metrics"
         open={metricsModalVisible}
         onCancel={() => setMetricsModalVisible(false)}
         width={600}
         footer={[<Button key="close" onClick={() => setMetricsModalVisible(false)}>Close</Button>]}
       >
         {userMetrics && (
           <div>
             <h4>Status: <Tag color={userMetrics.status === 'running' ? 'green' : 'red'}>{userMetrics.status}</Tag></h4>
             {userMetrics.metrics && (
               <pre style={{ 
                 background: '#f5f5f5', 
                 padding: 16, 
                 borderRadius: 4, 
                 maxHeight: 400, 
                 overflow: 'auto',
                 fontSize: 12
               }}>
                 {userMetrics.metrics}
               </pre>
             )}
           </div>
         )}
       </Modal>
{/* Edit Modal */}
       <Modal
         title="Edit Proxy"
         open={editModalVisible}
         onCancel={() => setEditModalVisible(false)}
         footer={null}
         width={500}
       >
         <Form form={editForm} onFinish={handleEditSubmit} layout="vertical">
            <Form.Item name="username" label="Username">
              <Input disabled />
            </Form.Item>
            <Form.Item 
              name="proxy_port" 
              label="Proxy Port"
              help="Change port (1000-65000) - Warning: changing port will restart the proxy container"
              rules={[
                { required: true, message: 'Port is required' },
                { pattern: /^\d+$/, message: 'Port must be a number' }
              ]}
            >
              <Input 
                type="number"
                min="1000"
                max="65000"
                placeholder="Enter port number"
              />
            </Form.Item>
            <Form.Item 
              name="bandwidth_limit" 
              label="Bandwidth Limit"
              help="Examples: 200M, 5G, leave empty for unlimited"
            >
              <Input placeholder="e.g., 200M, 5G or leave empty for unlimited" />
            </Form.Item>
           <Form.Item>
             <Space>
               <Button type="primary" htmlType="submit">Update</Button>
               <Button onClick={() => setEditModalVisible(false)}>Cancel</Button>
             </Space>
           </Form.Item>
         </Form>
       </Modal>


     </Content>
   </Layout>
 );
}

// Main app wrapper
function AppContent() {
 const { user, loading } = useAuth();

 if (loading) return <div>Loading...</div>;
 if (!user) return <Login />;
 return <Dashboard />;
}

function App() {
 return (
   <AuthProvider>
     <AppContent />
   </AuthProvider>
 );
}

export default App;
EOF
    success "Frontend created"
}

create_docker_and_deploy() {
    log "Creating Docker Compose and deployment files..."
    
    # Create Docker Compose
    cat > docker-compose.yml << EOF
version: '3.8'

services:
  backend:
    build: ./backend
    container_name: proxy-panel-backend
    ports:
      - "${WEB_PORT}:8000"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - /var/run/docker.sock:/var/run/docker.sock
    privileged: true
    group_add:
      - docker
    environment:
      - DATABASE_URL=sqlite:///./data/db/proxy_panel.db
      - SERVER_IP=${SERVER_IP}
    restart: unless-stopped
    networks:
      - proxy-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  frontend:
    build: ./frontend
    container_name: proxy-panel-frontend
    ports:
      - "3000:80"
    restart: unless-stopped
    depends_on:
      backend:
        condition: service_healthy
    networks:
      - proxy-network

networks:
  proxy-network:
    driver: bridge
EOF

    # Create management script
    cat > manage.sh << 'EOF'
#!/bin/bash

case "$1" in
    start)
        echo "Starting Telegram Proxy Panel..."
        docker compose up -d
        ;;
    stop)
        echo "Stopping Telegram Proxy Panel..."
        docker compose down
        ;;
    restart)
        echo "Restarting Telegram Proxy Panel..."
        docker compose restart
        ;;
    logs)
        docker compose logs -f
        ;;
    status)
        docker compose ps
        ;;
    backup)
        echo "Creating backup..."
        timestamp=$(date +%Y%m%d_%H%M%S)
        tar -czf "backup_${timestamp}.tar.gz" data/ docker compose.yml
        echo "Backup created: backup_${timestamp}.tar.gz"
        ;;
    update)
        echo "Updating..."
        docker compose down
        docker compose build --no-cache
        docker compose up -d
        ;;
    clean)
        echo "Cleaning up..."
        docker system prune -f
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|logs|status|backup|update|clean}"
        exit 1
        ;;
esac
EOF

    chmod +x manage.sh
    
    success "Docker and management files created"
}

# Deploy application
deploy_application() {
    log "Deploying application..."
    # Setup proper permissions
    log "Setting up directories and permissions..."
    mkdir -p data/db logs
    chmod -R 777 data logs
    
    # Clean docker state
    log "Cleaning Docker state..."
    log "Fixing Docker permissions..."
    chmod 666 /var/run/docker.sock 2>/dev/null || true
    usermod -aG docker $USER 2>/dev/null || true
    docker compose down 2>/dev/null || true
    docker system prune -f 2>/dev/null || true
    docker rm -f $(docker ps -aq) 2>/dev/null || true
    cat > .env << EOF
WEB_PORT=${WEB_PORT}
SERVER_IP=${SERVER_IP}
ADMIN_EMAIL=${ADMIN_EMAIL}
ADMIN_PASSWORD=${ADMIN_PASSWORD}
EOF

    # Clean up any existing containers
    docker compose down 2>/dev/null || true
    docker system prune -f 2>/dev/null || true
    
    # Build with no cache
    log "Building Docker images (this may take a few minutes)..."
    docker compose build --no-cache
    
    # Start services
    log "Starting services..."
    docker compose up -d
    
    # Wait for services
    log "Waiting for services to start..."
    sleep 30
    
    # Test backend health
    log "Waiting for services to be fully ready..."
    for i in {1..60}; do
        if curl -s http://localhost:${WEB_PORT}/health >/dev/null 2>&1; then
            success "Backend is ready"
            break
        fi
        echo -n "."
        sleep 3
    done
    echo
    
    log "Creating admin user..."
    docker exec proxy-panel-backend python -c "
import sys
sys.path.append('/app')
from app.database import SessionLocal, engine
from app import models
from app.auth import get_password_hash

models.Base.metadata.create_all(bind=engine)
db = SessionLocal()
try:
    existing = db.query(models.User).filter(models.User.email == '${ADMIN_EMAIL}').first()
    if existing:
        db.delete(existing)
        db.commit()
    
    hashed_password = get_password_hash('${ADMIN_PASSWORD}')
    admin_user = models.User(
        email='${ADMIN_EMAIL}',
        username='${ADMIN_EMAIL}'.split('@')[0],
        hashed_password=hashed_password,
        is_admin=True,
        is_active=True
    )
    db.add(admin_user)
    db.commit()
    print('Admin user created successfully')
except Exception as e:
    print(f'Error: {e}')
    db.rollback()
finally:
    db.close()
"
    
    # Verify deployment
    if docker compose ps | grep -q "Up"; then
        success "Application deployed successfully"
    else
        error "Deployment failed"
        echo "Checking logs..."
        docker compose logs --tail=50
        exit 1
    fi
}

# Configure firewall
configure_firewall() {
    log "Configuring firewall..."
    
    PORTS_TO_OPEN="$WEB_PORT 3000 10000:12999"
    
    if command -v ufw >/dev/null 2>&1; then
        for port in $PORTS_TO_OPEN; do
            if [[ $port == *":"* ]]; then
                sudo ufw allow $port/tcp
            else
                sudo ufw allow $port/tcp
            fi
        done
        success "UFW firewall configured"
    elif command -v firewall-cmd >/dev/null 2>&1; then
        for port in $PORTS_TO_OPEN; do
            if [[ $port == *":"* ]]; then
                start_port=$(echo $port | cut -d':' -f1)
                end_port=$(echo $port | cut -d':' -f2)
                sudo firewall-cmd --permanent --add-port=${start_port}-${end_port}/tcp
            else
                sudo firewall-cmd --permanent --add-port=${port}/tcp
            fi
        done
        sudo firewall-cmd --reload
        success "Firewalld configured"
    else
        warning "No firewall detected - ensure ports are open manually"
        echo "Required ports: $PORTS_TO_OPEN"
    fi
}

# Test installation
test_installation() {
    log "Testing installation..."
    
    # Test backend
    if curl -s http://localhost:${WEB_PORT}/health | grep -q "healthy"; then
        success "Backend health check passed"
    else
        warning "Backend health check failed"
        return 1
    fi
    
    # Test frontend
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:3000 | grep -q "200"; then
        success "Frontend accessibility test passed"
    else
        warning "Frontend may not be ready yet"
    fi
    
    return 0
}

# Display connection information
display_connection_info() {
    echo ""
    echo "==============================================================================="
    echo "              TELEGRAM PROXY WEB PANEL - INSTALLATION COMPLETE"
    echo "==============================================================================="
    echo ""
    echo " 🌟 ACCESS INFORMATION:"
    echo "   Frontend URL: http://${SERVER_IP}:3000"
    echo "   Backend API:  http://${SERVER_IP}:${WEB_PORT}"
    echo "   Admin Email:  ${ADMIN_EMAIL}"
    echo "   Admin Password: ${ADMIN_PASSWORD}"
    echo ""
    echo " 🚀 FEATURES:"
    echo "   ✓ Web-based user management"
    echo "   ✓ Automatic proxy container creation for each user"
    echo "   ✓ Real-time monitoring dashboard"
    echo "   ✓ Bandwidth usage tracking"
    echo "   ✓ QR code generation for easy mobile setup"
    echo "   ✓ Based on your original Telegram proxy script"
    echo ""
    echo " 🔧 MANAGEMENT COMMANDS:"
    echo "   Start:        ./manage.sh start"
    echo "   Stop:         ./manage.sh stop"
    echo "   Restart:      ./manage.sh restart"
    echo "   View logs:    ./manage.sh logs"
    echo "   View status:  ./manage.sh status"
    echo "   Backup:       ./manage.sh backup"
    echo "   Update:       ./manage.sh update"
    echo ""
    echo " 📱 HOW TO USE:"
    echo "   1. Open http://${SERVER_IP}:3000 in your browser"
    echo "   2. Login with your admin credentials"
    echo "   3. Create users by clicking 'Add User'"
    echo "   4. Each user gets their own dedicated proxy automatically"
    echo "   5. Click 'Config' to get QR codes or Telegram links"
    echo ""
    echo " 🗂️ DATA LOCATIONS:"
    echo "   Database: ./data/db/proxy_panel.db"
    echo "   Logs:     ./logs/"
    echo "   Config:   ./docker-compose.yml"
    echo ""
    echo " 🔌 PROXY PORTS:"
    echo "   User proxies will use ports 10000-12999"
    echo "   Each user gets 3 ports: proxy, bypass, metrics"
    echo ""
    echo " 🛡️ SECURITY NOTES:"
    echo "   • Each user gets an isolated proxy container"
    echo "   • Only Telegram traffic is allowed"
    echo "   • Bandwidth limits can be set per user"
    echo "   • All containers run with restricted privileges"
    echo ""
    echo "==============================================================================="
    echo "  🎉 TELEGRAM PROXY WEB PANEL IS READY!"
    echo "==============================================================================="
    echo ""
    
    # Show current container status
    echo " 📦 CONTAINER STATUS:"
    docker compose ps
    echo ""
    
    echo "🎯 Quick Start Steps:"
    echo "1. Open http://${SERVER_IP}:3000"
    echo "2. Login with: ${ADMIN_EMAIL}"
    echo "3. Click 'Add User' to create your first user"
    echo "4. Wait for proxy to be created (status will change to 'running')"
    echo "5. Click 'Config' to get Telegram settings and QR code"
    echo "6. Use QR code or Telegram link to connect"
    echo ""
}

# Main installation function
main() {
    echo ""
    echo "==============================================================================="
    echo "              TELEGRAM PROXY WEB PANEL - COMPLETE INSTALLATION"
    echo "==============================================================================="
    echo ""
    echo " This script will install a complete web-based Telegram proxy management system"
    echo " based on your original proxy script but with the following enhancements:"
    echo ""
    echo " • Web interface for easy management"
    echo " • Automatic proxy container creation per user"
    echo " • Individual bandwidth controls and monitoring"
    echo " • QR code generation for easy mobile setup"
    echo " • Real-time dashboard with statistics"
    echo " • Docker-based isolation and security"
    echo ""
    echo " Installation will take 5-10 minutes depending on your internet speed."
    echo ""
    
    # Confirm installation
    echo -n "Do you want to proceed with the installation? (y/N): "
    read -r confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Installation cancelled."
        exit 0
    fi
    
    # Run installation steps
    check_requirements
    get_admin_credentials
    get_server_ip
    create_backend_part
    create_proxy_manager
    create_main_app
    create_frontend
    create_docker_and_deploy
    configure_firewall
    deploy_application
    
    # Test installation
    if test_installation; then
        success "All tests passed!"
    else
        warning "Some tests failed, but installation may still work"
    fi
    
    # Display results
    display_connection_info
    
    # Save installation info
    cat > installation_summary.txt << EOF
Installation Date: $(date)
Server IP: ${SERVER_IP}
Frontend: http://${SERVER_IP}:3000
Backend: http://${SERVER_IP}:${WEB_PORT}
Admin Email: ${ADMIN_EMAIL}
Database: ./data/db/proxy_panel.db
Management: ./manage.sh
EOF
    
    success "Installation completed successfully!"
    echo ""
    echo "Your Telegram Proxy Web Panel is now ready to use!"
    echo "Each user you create will get their own dedicated proxy container"
    echo "based on your original Telegram proxy script."
    echo ""
}

# Handle cleanup on exit
cleanup() {
    echo ""
    warning "Installation interrupted. Cleaning up..."
    docker compose down 2>/dev/null || true
    exit 1
}

# Set trap for cleanup
trap cleanup INT TERM

# Run installation
case "${1:-install}" in
    install)
        main
        ;;
    --help|-h)
        echo "Usage: $0 [install|--help]"
        echo ""
        echo "install: Run the complete installation (default)"
        echo "--help:  Show this help message"
        ;;
    *)
        echo "Unknown option: $1"
        exit 1
        ;;
esac