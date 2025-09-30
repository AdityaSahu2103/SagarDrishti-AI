# FloatChat ARGO Data Explorer - Deployment Guide

## Overview
This guide provides step-by-step instructions for deploying the redesigned FloatChat ARGO Data Explorer dashboard with enhanced UI, 3D visualizations, and modern design.

## Prerequisites
- Python 3.8 or higher
- All existing backend dependencies
- ARGO data files in the specified directory
- OpenAI API key (for multimodal features)

## Installation Steps

### 1. Install Enhanced Dependencies
```bash
# Install the new requirements
pip install -r requirements_redesigned.txt

# Or install individual packages
pip install streamlit-audiorec streamlit-option-menu streamlit-elements
```

### 2. Backend Configuration
Ensure your existing backend configuration is intact:
- `config.py` - Configuration settings
- `data_ingest.py` - Data ingestion pipeline
- `embedding_index.py` - Vector embeddings
- `rag_engine.py` - RAG system
- Database connections (if using PostgreSQL)

### 3. Running the Redesigned Dashboard

#### Option A: Direct Streamlit Run
```bash
# Navigate to the FloatchatAI directory
cd FloatchatAI

# Run the redesigned dashboard
streamlit run dashboard_redesigned.py --server.port 8501 --server.address 0.0.0.0
```

#### Option B: Using the existing batch file (modified)
Create a new batch file `run_redesigned_dashboard.bat`:
```batch
@echo off
cd /d "%~dp0"
echo Starting FloatChat ARGO Data Explorer - Redesigned...
streamlit run dashboard_redesigned.py --server.port 8501 --server.address 0.0.0.0
pause
```

#### Option C: Using PowerShell script
Create `run_redesigned_dashboard.ps1`:
```powershell
Set-Location $PSScriptRoot
Write-Host "Starting FloatChat ARGO Data Explorer - Redesigned..." -ForegroundColor Green
streamlit run dashboard_redesigned.py --server.port 8501 --server.address 0.0.0.0
```

### 4. Access the Dashboard
- Open your browser and navigate to: `http://localhost:8501`
- The redesigned interface will load with the matrix-style header
- Use the theme toggle (🌙/☀️) in the top-right corner to switch between dark and light themes

## Key Features of the Redesigned Dashboard

### 1. Modern UI Design
- **Matrix-style Header**: Retro monospace font with glowing effects
- **Dual Theme System**: Dark and light themes with smooth transitions
- **Google Sans Font**: Professional typography throughout
- **Center-focused Layout**: No sidebar, clean centered design

### 2. Enhanced Data Visualization
- **Floating Data Bubbles**: Animated bubbles showing key metrics
- **3D Globe**: Interactive 3D globe showing ARGO float locations
- **Advanced Charts**: Modern Plotly visualizations with ocean themes
- **Responsive Design**: Works on desktop and mobile devices

### 3. Improved User Experience
- **Enhanced Voice Input**: Large circular recording button
- **Better Navigation**: Clean tab-based interface
- **Professional Styling**: Ocean-themed colors and gradients
- **Smooth Animations**: CSS animations and transitions

### 4. ARGO Project Information
- **Project Showcase**: Dedicated section about the ARGO project
- **Video Placeholder**: Ready for YouTube video embedding
- **Statistics Display**: Key ARGO program metrics

## Configuration Options

### Theme Customization
The dashboard supports extensive theme customization through CSS variables:
- Modify colors in the `:root` and `[data-theme="dark"]` sections
- Adjust gradients, shadows, and animations
- Customize ocean-themed color palette

### 3D Globe Settings
- Globe projection type can be changed in `create_3d_globe()`
- Marker sizes and colors are customizable
- Trajectory lines can be enabled/disabled

### Voice Recording
- Requires `streamlit-audiorec` package
- Audio processing uses OpenAI Whisper API
- Recording duration and quality can be adjusted

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install --upgrade -r requirements_redesigned.txt
   ```

2. **Audio Recording Not Working**
   - Ensure `streamlit-audiorec` is installed
   - Check browser permissions for microphone access
   - Verify OpenAI API key is configured

3. **3D Globe Not Loading**
   - Ensure Plotly is updated to latest version
   - Check browser compatibility (modern browsers required)
   - Verify data is properly loaded

4. **Theme Not Switching**
   - Clear browser cache
   - Check for JavaScript errors in browser console
   - Ensure CSS is properly loaded

### Performance Optimization

1. **Large Datasets**
   - Use data sampling for initial load
   - Implement lazy loading for visualizations
   - Cache frequently accessed data

2. **Browser Performance**
   - Use hardware acceleration
   - Limit concurrent visualizations
   - Optimize image sizes

## Deployment Options

### Local Development
```bash
streamlit run dashboard_redesigned.py
```

### Production Deployment
```bash
# Using Streamlit Cloud
streamlit run dashboard_redesigned.py --server.port 8501 --server.address 0.0.0.0

# Using Docker (create Dockerfile)
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements_redesigned.txt
EXPOSE 8501
CMD ["streamlit", "run", "dashboard_redesigned.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
```

### Cloud Deployment
- **Streamlit Cloud**: Direct deployment from GitHub
- **Heroku**: Using Procfile and requirements
- **AWS/GCP/Azure**: Container-based deployment

## Security Considerations

1. **API Keys**: Store in environment variables, not in code
2. **Data Access**: Implement proper authentication if needed
3. **CORS**: Configure for cross-origin requests if required
4. **HTTPS**: Use SSL certificates in production

## Monitoring and Maintenance

1. **Logs**: Monitor Streamlit logs for errors
2. **Performance**: Track loading times and user interactions
3. **Updates**: Regularly update dependencies
4. **Backup**: Maintain data and configuration backups

## Support and Documentation

- **Streamlit Documentation**: https://docs.streamlit.io/
- **Plotly Documentation**: https://plotly.com/python/
- **ARGO Program**: https://argo.ucsd.edu/

## Future Enhancements

The redesigned dashboard is ready for additional features:
- Real-time data updates
- Advanced filtering options
- Export functionality
- User authentication
- Collaborative features
- Mobile app version

For technical support or feature requests, contact the development team.
