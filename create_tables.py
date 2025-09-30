from sqlalchemy import create_engine
from config import Config
from db import Base

cfg = Config()
cfg.validate_config()

if cfg.POSTGRES_URL:
    db_url = cfg.POSTGRES_URL
else:
    db_url = (
        f"postgresql://{cfg.POSTGRES_USER}:{cfg.POSTGRES_PASSWORD}"
        f"@{cfg.POSTGRES_HOST}:{cfg.POSTGRES_PORT}/{cfg.POSTGRES_DB}"
    )

print("👉 Using DB URL:", db_url)

engine = create_engine(db_url, echo=True)
print("🔧 Creating tables...")
Base.metadata.create_all(engine)
print("✅ Tables created (or already exist).")
