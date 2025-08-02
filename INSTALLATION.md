# 🛡️ Linux AI Agent - Installation Guide

## 📋 Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 18.04+, CentOS 7+, or similar)
- **Python**: 3.8 or higher
- **Memory**: 2GB RAM minimum (4GB+ recommended for AI features)
- **Storage**: 1GB free space
- **Network**: Internet connection for package installation

### System Dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git

# CentOS/RHEL/Fedora
sudo yum install -y python3 python3-pip git
# or for newer versions:
sudo dnf install -y python3 python3-pip git

# Optional: ClamAV for enhanced scanning
sudo apt install -y clamav clamav-daemon  # Ubuntu/Debian
sudo yum install -y clamav clamav-update  # CentOS/RHEL
```

## 🚀 Installation Options

### Option 1: Basic Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/linux-ai-agent.git
cd linux-ai-agent

# Set up virtual environment
python3 -m venv venv
source venv/bin/activate

# Install basic dependencies
pip install -e .

# Test installation
python demo.py
```

### Option 2: Full Installation with AI Features

```bash
# After basic setup
pip install -e ".[ai,enhanced]"

# Download additional AI models (optional)
python -m spacy download en_core_web_sm
```

### Option 3: Development Installation

```bash
# Install with development tools
pip install -e ".[dev,ai,enhanced]"

# Run tests to verify
make test-basic
```

### Option 4: Using Make (Recommended for Developers)

```bash
# Install system dependencies (requires sudo)
make install-system-deps

# Set up virtual environment
make setup-venv
source venv/bin/activate

# Install with all features
make install-all

# Check status
make status
```

## 🔧 Configuration

### Initial Setup

1. **Create Configuration File**:
```bash
# The application will create a default config.json on first run
python src/main.py
# Press Ctrl+C to exit after initialization
```

2. **Configure ClamAV** (if installed):
```bash
# Update virus definitions
sudo freshclam

# Start ClamAV daemon (optional)
sudo systemctl enable clamav-daemon
sudo systemctl start clamav-daemon
```

3. **Set Environment Variables** (optional):
```bash
export LAI_DEBUG=true
export LAI_LOG_LEVEL=DEBUG
export LAI_QUARANTINE_DIR=/custom/quarantine/path
```

### Configuration File

The application creates `config.json` with default settings:

```json
{
    "debug": false,
    "log_level": "INFO",
    "log_file": "logs/agent.log",
    "quarantine_dir": "quarantine",
    "data_dir": "data",
    "scanner": {
        "entropy_threshold": 7.5,
        "max_file_size": 104857600,
        "clamav_timeout": 30,
        "heuristic_enabled": true,
        "quarantine_enabled": true
    },
    "ai": {
        "intent_confidence_threshold": 0.5,
        "rl_learning_rate": 0.001,
        "rl_epsilon_decay": 0.995,
        "rl_epsilon_min": 0.01,
        "gemma_enabled": false,
        "enhanced_nlp_enabled": false
    },
    "monitor": {
        "disk_warning_threshold": 80.0,
        "disk_critical_threshold": 95.0,
        "monitor_interval": 60,
        "log_retention_days": 30
    }
}
```

## ✅ Verification

### Basic Functionality Test

```bash
# Run the demo
python demo.py

# Expected output:
# 🛡️ Linux AI Agent - Demo
# ✅ Created file: demo_file.txt
# ✅ Read file content: This is a demo file...
# ✅ System stats: CPU X.X%, Memory X.X%
# ✅ Heuristic scanner initialized
# 🚀 Demo completed successfully!
```

### Interactive CLI Test

```bash
# Start the CLI
python src/main.py

# Try these commands:
help
disk usage
scan file demo.py
exit
```

### Advanced Features Test

```bash
# Test AI features (requires AI dependencies)
python test_ai_fixes.py

# Test all improvements
python test_improvements.py
```

## 🐛 Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'X'**
   ```bash
   # Make sure virtual environment is activated
   source venv/bin/activate
   
   # Reinstall dependencies
   pip install -e ".[ai,enhanced]"
   ```

2. **Permission denied when accessing files**
   ```bash
   # Check file permissions
   ls -la /path/to/file
   
   # Run with appropriate permissions
   sudo python src/main.py  # Only if necessary
   ```

3. **ClamAV not found**
   ```bash
   # Install ClamAV
   make install-system-deps
   
   # Or use heuristic scanning only
   # (application will auto-fallback)
   ```

4. **Virtual environment issues**
   ```bash
   # Remove and recreate
   rm -rf venv
   make setup-venv
   source venv/bin/activate
   make install
   ```

### Dependency Conflicts

If you encounter dependency conflicts:

```bash
# Use specific versions
pip install -r requirements.txt --force-reinstall

# Or install without AI features first
pip install -e .
# Then add AI features later
pip install -e ".[ai]"
```

## 🔒 Security Considerations

1. **File Permissions**: Ensure the application has appropriate read/write permissions
2. **Quarantine Directory**: Set up secure quarantine location with restricted access
3. **Log Files**: Protect log files as they may contain sensitive information
4. **Network Access**: The application may need internet access for updates

## 📊 Performance Optimization

### For Large File Scanning
```bash
# Increase file size limits in config.json
{
    "scanner": {
        "max_file_size": 1073741824  // 1GB
    }
}
```

### For Resource-Constrained Systems
```bash
# Disable AI features
{
    "ai": {
        "enhanced_nlp_enabled": false,
        "gemma_enabled": false
    }
}
```

## 📞 Support

If you encounter issues:

1. Check the logs: `tail -f logs/agent.log`
2. Run diagnostics: `make status`
3. Enable debug mode: Set `LAI_DEBUG=true`
4. Review documentation in `docs/` directory
5. Open an issue on GitHub with:
   - Your OS and Python version
   - Complete error message
   - Steps to reproduce

## 🎯 Next Steps

After successful installation:

1. Read the [User Guide](USER_GUIDE.md)
2. Explore [AI Features](AI_IMPLEMENTATION_GUIDE.md)
3. Check [Configuration Options](CONFIGURATION.md)
4. Review [Security Best Practices](SECURITY.md)