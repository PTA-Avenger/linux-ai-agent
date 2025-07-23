# Gemini Gamma Training VM Cloud-Init Setup

This cloud-init script sets up an Oracle Linux 9 VM optimized for training the Gemini Gamma model with security hardening and ML/AI development tools.

## VM Specifications

- **OS**: Oracle Linux 9 (Image build: 2025.06.17-0)
- **Security**: Shielded instance
- **VM Shape**: Standard2.4
  - 4 OCPU cores
  - 60 GB memory
  - 4.1 Gbps network bandwidth

## Prerequisites

1. **SSH Key Pair**: You mentioned you've already generated SSH keys on your local machine
2. **Oracle Cloud Infrastructure (OCI) Account**: With appropriate permissions to create compute instances

## Setup Instructions

### Step 1: Update SSH Key

Before using the cloud-init script, you **must** replace the placeholder SSH key:

1. Open `cloud-init.yaml`
2. Find this line:
   ```yaml
   - ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQ... your-public-key-here
   ```
3. Replace it with your actual public SSH key content (usually found in `~/.ssh/id_rsa.pub`)

### Step 2: Create VM Instance

1. **Log into Oracle Cloud Console**
2. **Navigate to Compute > Instances**
3. **Click "Create Instance"**
4. **Configure the instance**:
   - **Name**: `gemini-gamma-training-vm`
   - **Image**: Oracle Linux 9 (2025.06.17-0)
   - **Shape**: Standard2.4 (4 OCPU, 60 GB RAM)
   - **Security**: Enable Shielded Instance
   - **Networking**: Configure VCN and subnet as needed
   - **SSH Keys**: You can skip this since it's handled by cloud-init

5. **In the "Advanced Options" section**:
   - Click "Show Advanced Options"
   - Go to the "Management" tab
   - In the "Cloud-init script" section, paste the entire contents of `cloud-init.yaml`

6. **Click "Create"**

### Step 3: Wait for Setup

The cloud-init process will:
- Install all necessary packages and dependencies
- Set up Python virtual environment with ML/AI libraries
- Configure security (firewall, fail2ban)
- Create project structure
- Optimize system for ML training
- Reboot the instance

This process typically takes 10-15 minutes.

### Step 4: Connect to Your VM

```bash
# Replace <VM_IP> with your instance's public IP
ssh aitrainer@<VM_IP>
```

### Step 5: Start Training Environment

```bash
# Activate the Python virtual environment
source ~/gemini_env/bin/activate

# Navigate to project directory
cd ~/projects/gemini-gamma

# Start Jupyter Lab (optional)
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

# Or run the training script template
python scripts/train_gemini.py
```

## What's Installed

### System Packages
- Development tools (gcc, make, cmake, etc.)
- Python 3 with development headers
- Git, vim, htop, tmux, screen
- Container runtime (Podman)
- Security tools (fail2ban, firewalld)
- System monitoring tools

### Python Libraries
- **Core ML/AI**: PyTorch, Transformers, Accelerate
- **Google AI**: google-generativeai, google-cloud-aiplatform
- **Scientific Computing**: NumPy, Pandas, SciPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Development**: Jupyter Lab, Black, Pytest
- **Monitoring**: Weights & Biases, TensorBoard

### Project Structure
```
~/projects/gemini-gamma/
├── data/          # Training data
├── models/        # Saved models
├── logs/          # Training logs
├── scripts/       # Training scripts
└── notebooks/     # Jupyter notebooks
```

## Security Features

- **Firewall**: Configured with SSH (22) and Jupyter (8888) access
- **Fail2ban**: SSH brute-force protection
- **User Security**: Non-root user with sudo access
- **System Hardening**: Optimized kernel parameters and file limits

## System Optimizations

- **Memory Management**: 8GB swap file, optimized swappiness
- **Network**: Tuned TCP buffers for high-bandwidth training
- **File System**: Increased file descriptor limits
- **Log Rotation**: Automatic cleanup of training logs

## Troubleshooting

### Check Cloud-Init Status
```bash
# Check if cloud-init is complete
sudo cloud-init status

# View cloud-init logs
sudo cat /var/log/cloud-init-output.log
```

### Verify Python Environment
```bash
# Check if virtual environment exists
ls -la ~/gemini_env/

# Test Python installation
source ~/gemini_env/bin/activate
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

### Monitor System Resources
```bash
# Check memory usage
free -h

# Check CPU usage
htop

# Check disk space
df -h
```

## Customization

You can modify the cloud-init script to:
- Add additional Python packages to `requirements.txt`
- Change system optimization parameters
- Add custom configuration files
- Install additional software packages

## Cost Optimization Tips

1. **Stop the instance** when not training to avoid compute charges
2. **Use spot instances** for cost savings (if acceptable for your workload)
3. **Monitor resource usage** to ensure you're using the appropriate VM shape
4. **Set up billing alerts** to track costs

## Next Steps

1. Upload your training data to the `~/projects/gemini-gamma/data/` directory
2. Modify the training script template in `~/projects/gemini-gamma/scripts/train_gemini.py`
3. Configure your Google AI API credentials for Gemini access
4. Start your model training!

## Support

If you encounter issues:
1. Check the cloud-init logs: `sudo cat /var/log/cloud-init-output.log`
2. Verify all services are running: `systemctl status fail2ban firewalld`
3. Test network connectivity and firewall rules
4. Ensure your SSH key is correctly configured
