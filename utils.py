"""
Utility functions for ARGO oceanographic data processing and analysis.
Provides logging, validation, file operations, and helper functions.
"""

import logging
import os
import sys
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import warnings
import hashlib
import requests
from urllib.parse import urlparse
import time
from functools import wraps
import cftime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class ColoredFormatter(logging.Formatter):
    """Custom colored formatter for console logging."""
    
    # Color codes for different log levels
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Add color to levelname
        levelname_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{levelname_color}{record.levelname}{self.RESET}"
        
        # Add color to logger name
        record.name = f"\033[34m{record.name}{self.RESET}"
        
        return super().format(record)

def setup_logging(
    name: str = __name__, 
    level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Setup logging with colored console output and optional file logging.
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        console_output: Whether to output to console
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Set logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    # Create formatter
    formatter = ColoredFormatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Create log directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

class Timer:
    """Context manager for timing code execution."""
    
    def __init__(self, name: str = "Operation", logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.debug(f"Starting {self.name}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        self.logger.info(f"{self.name} completed in {duration:.2f} seconds")
        
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

def timing_decorator(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.debug(f"{func.__name__} executed in {duration:.2f} seconds")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func.__name__} failed after {duration:.2f} seconds: {e}")
            raise
    
    return wrapper

class FileManager:
    """File management utilities for ARGO data processing."""
    
    def __init__(self, base_directory: str = ".", logger: Optional[logging.Logger] = None):
        self.base_directory = Path(base_directory)
        self.logger = logger or setup_logging(self.__class__.__name__)
        
        # Create base directory if it doesn't exist
        self.base_directory.mkdir(parents=True, exist_ok=True)
    
    def ensure_directory(self, directory: Union[str, Path]) -> Path:
        """Ensure directory exists, create if necessary."""
        dir_path = Path(directory)
        if not dir_path.is_absolute():
            dir_path = self.base_directory / dir_path
        
        dir_path.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"Directory ensured: {dir_path}")
        return dir_path
    
    def list_files(self, directory: Union[str, Path], pattern: str = "*") -> List[Path]:
        """List files matching pattern in directory."""
        dir_path = Path(directory)
        if not dir_path.is_absolute():
            dir_path = self.base_directory / dir_path
        
        if not dir_path.exists():
            self.logger.warning(f"Directory does not exist: {dir_path}")
            return []
        
        files = list(dir_path.glob(pattern))
        self.logger.debug(f"Found {len(files)} files matching '{pattern}' in {dir_path}")
        return files
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Get comprehensive file information."""
        path = Path(file_path)
        
        if not path.exists():
            return {"exists": False}
        
        stat = path.stat()
        
        return {
            "exists": True,
            "path": str(path.absolute()),
            "name": path.name,
            "suffix": path.suffix,
            "size_bytes": stat.st_size,
            "size_mb": stat.st_size / (1024 * 1024),
            "created": datetime.fromtimestamp(stat.st_ctime),
            "modified": datetime.fromtimestamp(stat.st_mtime),
            "is_file": path.is_file(),
            "is_directory": path.is_dir()
        }
    
    def safe_remove(self, file_path: Union[str, Path]) -> bool:
        """Safely remove file with error handling."""
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()
                self.logger.debug(f"Removed file: {path}")
                return True
            else:
                self.logger.warning(f"File does not exist: {path}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to remove file {file_path}: {e}")
            return False
    
    def create_backup(self, file_path: Union[str, Path]) -> Optional[Path]:
        """Create backup copy of file."""
        try:
            original = Path(file_path)
            if not original.exists():
                self.logger.error(f"Cannot backup non-existent file: {original}")
                return None
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = original.with_suffix(f".{timestamp}{original.suffix}")
            
            import shutil
            shutil.copy2(original, backup_path)
            
            self.logger.info(f"Created backup: {backup_path}")
            return backup_path
            
        except Exception as e:
            self.logger.error(f"Failed to create backup for {file_path}: {e}")
            return None

class DataValidator:
    """Data validation utilities for ARGO profiles."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or setup_logging(self.__class__.__name__)
    
    def validate_coordinates(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """Validate geographic coordinates."""
        issues = []
        
        # Latitude validation
        if not -90 <= latitude <= 90:
            issues.append(f"Invalid latitude: {latitude} (must be between -90 and 90)")
        
        # Longitude validation  
        if not -180 <= longitude <= 180:
            issues.append(f"Invalid longitude: {longitude} (must be between -180 and 180)")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "coordinates": (latitude, longitude)
        }
    
    def validate_oceanographic_data(self, 
                                  temperature: np.ndarray, 
                                  salinity: np.ndarray, 
                                  pressure: np.ndarray) -> Dict[str, Any]:
        """Validate oceanographic measurement arrays."""
        issues = []
        
        # Check array lengths
        if not (len(temperature) == len(salinity) == len(pressure)):
            issues.append("Temperature, salinity, and pressure arrays must have same length")
        
        # Temperature validation (typical ocean range: -2 to 40°C)
        temp_min, temp_max = -2.0, 40.0
        invalid_temp = (temperature < temp_min) | (temperature > temp_max)
        if np.any(invalid_temp & ~np.isnan(temperature)):
            issues.append(f"Temperature values outside expected range ({temp_min}-{temp_max}°C)")
        
        # Salinity validation (typical ocean range: 30 to 42 PSU)
        sal_min, sal_max = 30.0, 42.0
        invalid_sal = (salinity < sal_min) | (salinity > sal_max)
        if np.any(invalid_sal & ~np.isnan(salinity)):
            issues.append(f"Salinity values outside expected range ({sal_min}-{sal_max} PSU)")
        
        # Pressure validation (should be positive and increasing with depth)
        if np.any(pressure < 0):
            issues.append("Negative pressure values found")
        
        # Check for monotonically increasing pressure (allowing for small variations)
        pressure_diff = np.diff(pressure[~np.isnan(pressure)])
        if np.any(pressure_diff < -5):  # Allow small decreases due to measurement noise
            issues.append("Pressure not monotonically increasing (potential depth ordering issue)")
        
        # Check for excessive NaN values
        nan_threshold = 0.5  # 50% threshold
        for name, data in [("temperature", temperature), ("salinity", salinity), ("pressure", pressure)]:
            nan_fraction = np.sum(np.isnan(data)) / len(data)
            if nan_fraction > nan_threshold:
                issues.append(f"Excessive NaN values in {name}: {nan_fraction:.1%}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "statistics": {
                "temperature": self._get_array_stats(temperature),
                "salinity": self._get_array_stats(salinity), 
                "pressure": self._get_array_stats(pressure)
            }
        }
    
    def _get_array_stats(self, data: np.ndarray) -> Dict[str, float]:
        """Get basic statistics for an array."""
        valid_data = data[~np.isnan(data)]
        if len(valid_data) == 0:
            return {"count": 0, "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
        
        return {
            "count": len(valid_data),
            "mean": np.mean(valid_data),
            "std": np.std(valid_data),
            "min": np.min(valid_data),
            "max": np.max(valid_data)
        }
    
    def validate_argo_dataset(self, dataset: xr.Dataset) -> Dict[str, Any]:
        """Validate an ARGO xarray Dataset."""
        issues = []
        
        # Check for required variables
        required_vars = ['TEMP', 'PSAL', 'PRES']
        alternative_vars = {
            'TEMP': ['temp', 'temperature'],
            'PSAL': ['psal', 'salinity'], 
            'PRES': ['pres', 'pressure']
        }
        
        available_vars = {}
        for var in required_vars:
            if var in dataset.data_vars:
                available_vars[var] = var
            else:
                # Check for alternative names
                found = False
                for alt_var in alternative_vars.get(var, []):
                    if alt_var in dataset.data_vars:
                        available_vars[var] = alt_var
                        found = True
                        break
                
                if not found:
                    issues.append(f"Missing required variable: {var}")
        
        # Check for coordinate variables
        coord_vars = ['LATITUDE', 'LONGITUDE']
        coord_alternatives = {
            'LATITUDE': ['latitude', 'lat'],
            'LONGITUDE': ['longitude', 'lon', 'long']
        }
        
        for coord in coord_vars:
            if coord not in dataset.coords and coord.lower() not in dataset.coords:
                found = False
                for alt_coord in coord_alternatives.get(coord, []):
                    if alt_coord in dataset.coords:
                        found = True
                        break
                
                if not found:
                    issues.append(f"Missing coordinate: {coord}")
        
        # Check data dimensions
        if available_vars:
            first_var = list(available_vars.values())[0]
            expected_dims = dataset[first_var].dims
            
            for var_name, dataset_var in available_vars.items():
                if dataset[dataset_var].dims != expected_dims:
                    issues.append(f"Dimension mismatch in {var_name}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "available_variables": available_vars,
            "dataset_info": {
                "dimensions": dict(dataset.dims),
                "coordinates": list(dataset.coords),
                "data_variables": list(dataset.data_vars)
            }
        }

class CacheManager:
    """Simple caching system for processed data."""
    
    def __init__(self, cache_dir: str = "cache", logger: Optional[logging.Logger] = None):
        self.cache_dir = Path(cache_dir)
        self.logger = logger or setup_logging(self.__class__.__name__)
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, key: str) -> Path:
        """Generate cache file path from key."""
        # Create hash of key for filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"
    
    def set(self, key: str, data: Any, ttl_hours: int = 24) -> bool:
        """Store data in cache with TTL."""
        try:
            cache_path = self._get_cache_path(key)
            
            cache_data = {
                "data": data,
                "timestamp": datetime.now(),
                "ttl_hours": ttl_hours,
                "key": key
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            self.logger.debug(f"Cached data for key: {key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cache data for key {key}: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve data from cache if valid."""
        try:
            cache_path = self._get_cache_path(key)
            
            if not cache_path.exists():
                return None
            
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Check TTL
            age_hours = (datetime.now() - cache_data["timestamp"]).total_seconds() / 3600
            if age_hours > cache_data["ttl_hours"]:
                self.logger.debug(f"Cache expired for key: {key}")
                cache_path.unlink()  # Remove expired cache
                return None
            
            self.logger.debug(f"Cache hit for key: {key}")
            return cache_data["data"]
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve cache for key {key}: {e}")
            return None
    
    def clear(self, key: Optional[str] = None) -> int:
        """Clear cache entries. If key is None, clear all."""
        count = 0
        
        if key is not None:
            # Clear specific key
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                cache_path.unlink()
                count = 1
                self.logger.debug(f"Cleared cache for key: {key}")
        else:
            # Clear all cache files
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
                count += 1
            
            self.logger.info(f"Cleared {count} cache entries")
        
        return count
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cache contents."""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "cache_directory": str(self.cache_dir),
            "total_files": len(cache_files),
            "total_size_mb": total_size / (1024 * 1024),
            "cache_files": [f.name for f in cache_files]
        }

class ConfigManager:
    """Configuration management utility."""
    
    def __init__(self, config_file: str = "config.json", logger: Optional[logging.Logger] = None):
        self.config_file = Path(config_file)
        self.logger = logger or setup_logging(self.__class__.__name__)
        self._config = {}
        self.load()
    
    def load(self) -> bool:
        """Load configuration from file."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    self._config = json.load(f)
                self.logger.debug(f"Loaded configuration from {self.config_file}")
                return True
            else:
                self.logger.info(f"Configuration file {self.config_file} not found, using defaults")
                return False
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            return False
    
    def save(self) -> bool:
        """Save current configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self._config, f, indent=2, default=str)
            self.logger.debug(f"Saved configuration to {self.config_file}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        keys = key.split('.')
        config = self._config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
        self.logger.debug(f"Set configuration {key} = {value}")
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values."""
        return self._config.copy()
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple configuration values."""
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self._config, updates)
        self.logger.debug(f"Updated configuration with {len(updates)} values")

def convert_cftime_to_datetime(time_value: Any) -> datetime:
    """Convert various time formats to datetime."""
    if isinstance(time_value, datetime):
        return time_value
    elif isinstance(time_value, cftime.Datetime360Day):
        return pd.Timestamp(time_value.isoformat()).to_pydatetime()
    elif isinstance(time_value, np.datetime64):
        return pd.Timestamp(time_value).to_pydatetime()
    elif isinstance(time_value, (int, float)):
        # Assume days since 1900-01-01 (common ARGO format)
        base_date = datetime(1900, 1, 1)
        return base_date + timedelta(days=time_value)
    else:
        raise ValueError(f"Unsupported time format: {type(time_value)}")

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points using Haversine formula.
    Returns distance in kilometers.
    """
    # Convert latitude and longitude to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Earth's radius in kilometers
    r = 6371
    
    return c * r

def create_profile_summary(profile_data: Dict[str, Any]) -> str:
    """Create human-readable summary of an ARGO profile."""
    try:
        lat = profile_data.get('latitude', 0)
        lon = profile_data.get('longitude', 0)
        
        # Calculate mean values
        temp_data = np.array(profile_data.get('temperature', []))
        sal_data = np.array(profile_data.get('salinity', []))
        pres_data = np.array(profile_data.get('pressure', []))
        
        temp_mean = np.nanmean(temp_data) if len(temp_data) > 0 else np.nan
        sal_mean = np.nanmean(sal_data) if len(sal_data) > 0 else np.nan
        max_depth = np.nanmax(pres_data) if len(pres_data) > 0 else np.nan
        
        # Get time information
        profile_time = profile_data.get('time', datetime.now())
        if not isinstance(profile_time, datetime):
            profile_time = convert_cftime_to_datetime(profile_time)
        
        # Create summary text
        summary_parts = []
        summary_parts.append(f"ARGO profile at {lat:.2f}°N, {lon:.2f}°E")
        summary_parts.append(f"measured on {profile_time.strftime('%Y-%m-%d')}")
        
        if not np.isnan(temp_mean):
            summary_parts.append(f"mean temperature {temp_mean:.2f}°C")
        
        if not np.isnan(sal_mean):
            summary_parts.append(f"mean salinity {sal_mean:.2f} PSU")
        
        if not np.isnan(max_depth):
            summary_parts.append(f"maximum depth {max_depth:.0f}m")
        
        return ", ".join(summary_parts)
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Error creating profile summary: {e}")
        return f"ARGO profile at {lat:.2f}°N, {lon:.2f}°E"

def download_file(url: str, destination: Union[str, Path], chunk_size: int = 8192) -> bool:
    """
    Download file from URL with progress tracking.
    
    Args:
        url: URL to download from
        destination: Local path to save file
        chunk_size: Download chunk size in bytes
        
    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        logger.debug(f"Download progress: {progress:.1f}%")
        
        logger.info(f"Downloaded {url} to {destination}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_names[i]}"

def get_system_info() -> Dict[str, Any]:
    """Get system information for debugging."""
    import platform
    import psutil
    
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_total_gb": psutil.virtual_memory().total / (1024**3),
        "memory_available_gb": psutil.virtual_memory().available / (1024**3),
        "disk_usage": {
            "total_gb": psutil.disk_usage('/').total / (1024**3),
            "free_gb": psutil.disk_usage('/').free / (1024**3)
        }
    }

# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logger = setup_logging(__name__, level="DEBUG")
    logger.info("Testing utility functions...")
    
    # Test timer
    with Timer("Test operation", logger):
        time.sleep(1)
    
    # Test file manager
    fm = FileManager()
    test_dir = fm.ensure_directory("test_data")
    logger.info(f"Created test directory: {test_dir}")
    
    # Test cache manager
    cache = CacheManager()
    cache.set("test_key", {"data": "test_value"})
    cached_data = cache.get("test_key")
    logger.info(f"Cache test result: {cached_data}")
    
    # Test configuration manager
    config = ConfigManager("test_config.json")
    config.set("database.host", "localhost")
    config.set("database.port", 5432)
    config.save()
    
    logger.info(f"Configuration test: {config.get_all()}")
    
    # Test system info
    sys_info = get_system_info()
    logger.info(f"System info: {sys_info}")
    
    logger.info("Utility functions test completed!")