"""
Authentication and Security Middleware for OpenManus API
JWT-based authentication with role-based access control
"""

import os
import jwt
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from functools import wraps

from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from pydantic import BaseModel


# Configuration
class AuthConfig:
    """Authentication configuration"""
    def __init__(self):
        self.jwt_secret = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
        self.jwt_algorithm = "HS256"
        self.access_token_expire_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
        self.refresh_token_expire_days = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
        self.password_min_length = int(os.getenv("PASSWORD_MIN_LENGTH", "8"))
        self.enable_registration = os.getenv("ENABLE_REGISTRATION", "true").lower() == "true"
        self.require_email_verification = os.getenv("REQUIRE_EMAIL_VERIFICATION", "false").lower() == "true"


# Models
class UserRole(str):
    """User roles"""
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"


class User(BaseModel):
    """User model"""
    id: str
    username: str
    email: str
    role: UserRole = UserRole.USER
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime
    last_login: Optional[datetime] = None
    metadata: Dict[str, Any] = {}


class UserCreate(BaseModel):
    """User creation model"""
    username: str
    email: str
    password: str
    role: UserRole = UserRole.USER


class UserLogin(BaseModel):
    """User login model"""
    username: str
    password: str


class Token(BaseModel):
    """Token model"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """Token data model"""
    user_id: str
    username: str
    role: UserRole
    exp: datetime


# Security utilities
class SecurityManager:
    """Security manager for authentication and authorization"""
    
    def __init__(self, config: AuthConfig = None):
        self.config = config or AuthConfig()
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.security = HTTPBearer(auto_error=False)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # In-memory user store (replace with database in production)
        self.users: Dict[str, User] = {}
        self.user_passwords: Dict[str, str] = {}
        
        # Create default admin user
        self._create_default_admin()
    
    def _create_default_admin(self):
        """Create default admin user"""
        admin_username = os.getenv("ADMIN_USERNAME", "admin")
        admin_password = os.getenv("ADMIN_PASSWORD", "admin123")
        admin_email = os.getenv("ADMIN_EMAIL", "admin@openmanus.ai")
        
        if admin_username not in self.users:
            admin_user = User(
                id="admin-001",
                username=admin_username,
                email=admin_email,
                role=UserRole.ADMIN,
                is_active=True,
                is_verified=True,
                created_at=datetime.now()
            )
            
            self.users[admin_username] = admin_user
            self.user_passwords[admin_username] = self.hash_password(admin_password)
            
            self.logger.info(f"Default admin user created: {admin_username}")
    
    def hash_password(self, password: str) -> str:
        """Hash password"""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(self, user: User) -> str:
        """Create access token"""
        expire = datetime.utcnow() + timedelta(minutes=self.config.access_token_expire_minutes)
        
        to_encode = {
            "user_id": user.id,
            "username": user.username,
            "role": user.role,
            "exp": expire
        }
        
        return jwt.encode(to_encode, self.config.jwt_secret, algorithm=self.config.jwt_algorithm)
    
    def create_refresh_token(self, user: User) -> str:
        """Create refresh token"""
        expire = datetime.utcnow() + timedelta(days=self.config.refresh_token_expire_days)
        
        to_encode = {
            "user_id": user.id,
            "username": user.username,
            "type": "refresh",
            "exp": expire
        }
        
        return jwt.encode(to_encode, self.config.jwt_secret, algorithm=self.config.jwt_algorithm)
    
    def verify_token(self, token: str) -> Optional[TokenData]:
        """Verify and decode token"""
        try:
            payload = jwt.decode(token, self.config.jwt_secret, algorithms=[self.config.jwt_algorithm])
            
            user_id = payload.get("user_id")
            username = payload.get("username")
            role = payload.get("role")
            exp = datetime.fromtimestamp(payload.get("exp"))
            
            if not user_id or not username:
                return None
            
            return TokenData(
                user_id=user_id,
                username=username,
                role=UserRole(role) if role else UserRole.USER,
                exp=exp
            )
            
        except jwt.ExpiredSignatureError:
            self.logger.warning("Token has expired")
            return None
        except jwt.JWTError as e:
            self.logger.warning(f"Token verification failed: {e}")
            return None
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user"""
        user = self.users.get(username)
        if not user:
            return None
        
        if not user.is_active:
            return None
        
        stored_password = self.user_passwords.get(username)
        if not stored_password:
            return None
        
        if not self.verify_password(password, stored_password):
            return None
        
        # Update last login
        user.last_login = datetime.now()
        return user
    
    def create_user(self, user_data: UserCreate) -> User:
        """Create new user"""
        if not self.config.enable_registration:
            raise HTTPException(status_code=403, detail="Registration is disabled")
        
        if user_data.username in self.users:
            raise HTTPException(status_code=400, detail="Username already exists")
        
        # Validate password
        if len(user_data.password) < self.config.password_min_length:
            raise HTTPException(
                status_code=400, 
                detail=f"Password must be at least {self.config.password_min_length} characters"
            )
        
        # Create user
        user = User(
            id=f"user-{len(self.users) + 1:03d}",
            username=user_data.username,
            email=user_data.email,
            role=user_data.role,
            is_active=True,
            is_verified=not self.config.require_email_verification,
            created_at=datetime.now()
        )
        
        # Store user and password
        self.users[user_data.username] = user
        self.user_passwords[user_data.username] = self.hash_password(user_data.password)
        
        self.logger.info(f"User created: {user_data.username}")
        return user
    
    def get_user(self, username: str) -> Optional[User]:
        """Get user by username"""
        return self.users.get(username)
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        for user in self.users.values():
            if user.id == user_id:
                return user
        return None
    
    def update_user(self, username: str, updates: Dict[str, Any]) -> Optional[User]:
        """Update user"""
        user = self.users.get(username)
        if not user:
            return None
        
        # Update allowed fields
        allowed_fields = ["email", "role", "is_active", "is_verified", "metadata"]
        for field, value in updates.items():
            if field in allowed_fields and hasattr(user, field):
                setattr(user, field, value)
        
        return user
    
    def delete_user(self, username: str) -> bool:
        """Delete user"""
        if username in self.users:
            del self.users[username]
            if username in self.user_passwords:
                del self.user_passwords[username]
            self.logger.info(f"User deleted: {username}")
            return True
        return False
    
    def list_users(self) -> List[User]:
        """List all users"""
        return list(self.users.values())


# Global security manager
security_manager = SecurityManager()


# Dependency functions
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security_manager.security)) -> Optional[User]:
    """Get current authenticated user"""
    if not credentials:
        return None
    
    token_data = security_manager.verify_token(credentials.credentials)
    if not token_data:
        return None
    
    user = security_manager.get_user(token_data.username)
    if not user or not user.is_active:
        return None
    
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user (required)"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    
    return current_user


async def get_current_admin_user(current_user: User = Depends(get_current_active_user)) -> User:
    """Get current admin user (required)"""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    return current_user


# Role-based access control decorators
def require_role(required_role: UserRole):
    """Decorator to require specific role"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract current_user from kwargs
            current_user = kwargs.get('current_user')
            if not current_user:
                raise HTTPException(status_code=401, detail="Authentication required")
            
            if current_user.role != required_role and current_user.role != UserRole.ADMIN:
                raise HTTPException(status_code=403, detail=f"Role {required_role} required")
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def require_permission(permission: str):
    """Decorator to require specific permission"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract current_user from kwargs
            current_user = kwargs.get('current_user')
            if not current_user:
                raise HTTPException(status_code=401, detail="Authentication required")
            
            # Check permission (simplified - in production, use proper permission system)
            user_permissions = current_user.metadata.get("permissions", [])
            if permission not in user_permissions and current_user.role != UserRole.ADMIN:
                raise HTTPException(status_code=403, detail=f"Permission {permission} required")
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Rate limiting
class RateLimiter:
    """Simple rate limiter"""
    
    def __init__(self, max_requests: int = 60, window_minutes: int = 1):
        self.max_requests = max_requests
        self.window_minutes = window_minutes
        self.requests: Dict[str, List[datetime]] = {}
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed"""
        now = datetime.now()
        cutoff = now - timedelta(minutes=self.window_minutes)
        
        # Clean old requests
        if identifier in self.requests:
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if req_time > cutoff
            ]
        else:
            self.requests[identifier] = []
        
        # Check limit
        if len(self.requests[identifier]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[identifier].append(now)
        return True


# Global rate limiter
rate_limiter = RateLimiter()


async def check_rate_limit(request: Request):
    """Rate limiting dependency"""
    client_ip = request.client.host
    
    # Get forwarded IP if behind proxy
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()
    
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")


# Security headers middleware
async def add_security_headers(request: Request, call_next):
    """Add security headers to responses"""
    response = await call_next(request)
    
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    return response


# Utility functions
def get_password_hash(password: str) -> str:
    """Get password hash"""
    return security_manager.hash_password(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password"""
    return security_manager.verify_password(plain_password, hashed_password)


def create_tokens(user: User) -> Token:
    """Create access and refresh tokens"""
    access_token = security_manager.create_access_token(user)
    refresh_token = security_manager.create_refresh_token(user)
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=security_manager.config.access_token_expire_minutes * 60
    )


# Initialize security
def initialize_security(config: dict = None) -> SecurityManager:
    """Initialize security with configuration"""
    global security_manager
    
    if config:
        auth_config = AuthConfig()
        for key, value in config.items():
            if hasattr(auth_config, key):
                setattr(auth_config, key, value)
        
        security_manager = SecurityManager(auth_config)
    
    return security_manager

