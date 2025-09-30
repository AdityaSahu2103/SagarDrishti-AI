"""
ARGO NetCDF data ingestion module.
Downloads, parses, and processes ARGO float profiles from NOAA servers.
"""

import os
import urllib.request
import xarray as xr
import pandas as pd
import numpy as np
import cftime
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Tuple
from config import Config
from utils import setup_logging

logger = setup_logging(__name__)

class ARGODataIngestor:
    def __init__(self, config: Config = Config):
        self.config = config
        self.config.ensure_data_dirs()
    
    def download_netcdf_files(self, year: str = None, month: str = None, 
                             max_files: int = 10) -> List[str]:
        """
        Download ARGO NetCDF files from NOAA server.
        
        Args:
            year: Target year (default from config)
            month: Target month (default from config)
            max_files: Maximum number of files to download
            
        Returns:
            List of downloaded file paths
        """
        year = year or self.config.DEFAULT_YEAR
        month = month or self.config.DEFAULT_MONTH
        
        url = f"{self.config.ARGO_BASE_URL}/{year}/{month}"
        downloaded_files = []
        
        try:
            logger.info(f"Fetching directory listing from: {url}")
            response = urllib.request.urlopen(url)
            html = response.read()
            soup = BeautifulSoup(html, 'html.parser')
            
            links = soup.find_all('a')
            nc_links = [link.get('href') for link in links 
                       if link.get('href') and link.get('href').endswith('.nc')]
            
            logger.info(f"Found {len(nc_links)} NetCDF files")
            
            for i, nc_file in enumerate(nc_links[:max_files]):
                file_url = f"{url}/{nc_file}"
                local_path = os.path.join(self.config.INDIAN_OCEAN_PATH, nc_file)
                
                if os.path.exists(local_path):
                    logger.info(f"File already exists: {nc_file}")
                    downloaded_files.append(local_path)
                    continue
                
                logger.info(f"Downloading ({i+1}/{min(max_files, len(nc_links))}): {nc_file}")
                urllib.request.urlretrieve(file_url, local_path)
                downloaded_files.append(local_path)
                
        except Exception as e:
            logger.error(f"Error downloading files: {e}")
            
        return downloaded_files
    
    def parse_netcdf_profile(self, file_path: str, profile_idx: int = 0) -> Optional[pd.DataFrame]:
        """
        Parse a single NetCDF file into a pandas DataFrame.
        
        Args:
            file_path: Path to NetCDF file
            profile_idx: Index of profile to extract (default: 0)
            
        Returns:
            DataFrame with profile data or None if parsing fails
        """
        try:
            ds = xr.open_dataset(file_path)
            
            pressure_vars = ['pres', 'PRES', 'pressure']
            temp_vars = ['temp', 'TEMP', 'temperature']
            sal_vars = ['psal', 'PSAL', 'salinity']
            
            pressure = self._get_variable(ds, pressure_vars, profile_idx)
            temperature = self._get_variable(ds, temp_vars, profile_idx)
            salinity = self._get_variable(ds, sal_vars, profile_idx)
            
            if pressure is None or temperature is None:
                logger.warning(f"Missing required variables in {file_path}")
                return None
            
            latitude = ds['latitude'].values[profile_idx] if 'latitude' in ds else np.nan
            longitude = ds['longitude'].values[profile_idx] if 'longitude' in ds else np.nan
            
            time = self._extract_time(ds, profile_idx)
     
            n_levels = len(pressure)
            df = pd.DataFrame({
                'latitude': [latitude] * n_levels,
                'longitude': [longitude] * n_levels,
                'time': [time] * n_levels,
                'pressure': pressure,
                'temperature': temperature,
                'salinity': salinity if salinity is not None else [np.nan] * n_levels,
                'file_source': [os.path.basename(file_path)] * n_levels
            })
            
            df = df.dropna(subset=['pressure', 'temperature'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return None
    
    def _get_variable(self, ds: xr.Dataset, var_names: List[str], 
                     profile_idx: int) -> Optional[np.ndarray]:
        """Extract variable data, trying multiple possible names."""
        for var_name in var_names:
            if var_name in ds:
                return ds[var_name].isel(n_prof=profile_idx).values
        return None
    
    def _extract_time(self, ds: xr.Dataset, profile_idx: int) -> pd.Timestamp:
        """Extract and convert time data to pandas Timestamp."""
        time_vars = ['juld', 'JULD', 'time']
        
        for time_var in time_vars:
            if time_var in ds:
                time_val = ds[time_var].values[profile_idx]
                
                if isinstance(time_val, cftime.Datetime360Day):
                    return pd.Timestamp(time_val.isoformat())
                elif np.issubdtype(type(time_val), np.datetime64):
                    return pd.Timestamp(time_val)
                elif isinstance(time_val, (int, float)):
                    return pd.Timestamp('1950-01-01') + pd.Timedelta(days=float(time_val))
        
        # Default to current time if no time found
        return pd.Timestamp.now()
    
    def process_all_files(self) -> List[pd.DataFrame]:
        """
        Process all NetCDF files in the data directory.
        
        Returns:
            List of DataFrames, one per successfully parsed profile
        """
        nc_files = [f for f in os.listdir(self.config.INDIAN_OCEAN_PATH) 
                   if f.endswith('.nc')]
        
        if not nc_files:
            logger.warning("No NetCDF files found. Run download_netcdf_files() first.")
            return []
        
        profiles = []
        logger.info(f"Processing {len(nc_files)} NetCDF files...")
        
        for nc_file in nc_files:
            file_path = os.path.join(self.config.INDIAN_OCEAN_PATH, nc_file)
            df = self.parse_netcdf_profile(file_path)
            
            if df is not None and not df.empty:
                profiles.append(df)
            else:
                logger.warning(f"Skipped empty/invalid file: {nc_file}")
        
        logger.info(f"Successfully processed {len(profiles)} profiles")
        return profiles
    
    def get_file_list(self) -> List[str]:
        """Get list of available NetCDF files."""
        return [f for f in os.listdir(self.config.INDIAN_OCEAN_PATH) 
                if f.endswith('.nc')]

def create_profile_summary(df: pd.DataFrame) -> str:
    """Create human-readable summary of an ARGO profile."""
    if df.empty:
        return "Empty profile"
    
    lat = df['latitude'].iloc[0]
    lon = df['longitude'].iloc[0]
    month = df['time'].iloc[0].month
    temp_mean = df['temperature'].mean()
    sal_mean = df['salinity'].mean()
    
    return (f"Float near lat {lat:.2f}, lon {lon:.2f} "
           f"in month {month}, mean temp {temp_mean:.2f}°C, "
           f"mean salinity {sal_mean:.2f} PSU")

if __name__ == "__main__":
    # Initialize ingestor
    ingestor = ARGODataIngestor()
    
    downloaded = ingestor.download_netcdf_files(max_files=5)
    print(f"Downloaded {len(downloaded)} files")
    
    profiles = ingestor.process_all_files()
    
    for profile in profiles[:3]:  # Show first 3
        summary = create_profile_summary(profile)
        print(summary)