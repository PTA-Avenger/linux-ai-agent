# 🔮 **Gemma 3 4B Integration Setup Guide**

**Linux AI Agent - Advanced AI Capabilities**

---

## 🎯 **Overview**

The Linux AI Agent now supports **Gemma 3 4B integration** with multiple deployment modes:
- **API Mode**: Using Google AI Studio API
- **Local Mode**: Running Gemma locally with transformers
- **Custom Mode**: Custom API endpoints
- **Auto Mode**: Automatically selects the best available option

## 🚀 **Quick Start**

### **Option 1: API Mode (Recommended)**
```bash
# 1. Install dependencies
pip install google-generativeai

# 2. Get API key from Google AI Studio
# Visit: https://makersuite.google.com/app/apikey

# 3. Set environment variable
export GEMMA_API_KEY="your_api_key_here"

# 4. Start the agent
python3 src/main.py
```

### **Option 2: Local Mode**
```bash
# 1. Install dependencies (requires GPU for optimal performance)
pip install transformers torch accelerate

# 2. Start the agent (will download model on first use)
python3 src/main.py
```

---

## 📋 **Detailed Setup Instructions**

### **🔧 Installation**

#### **1. Install Core Dependencies**
```bash
# Basic requirements
pip install -r requirements.txt

# Gemma-specific dependencies
pip install google-generativeai transformers torch
```

#### **2. Optional Dependencies for Enhanced Features**
```bash
# For GPU acceleration (NVIDIA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU optimization
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For model quantization (reduces memory usage)
pip install bitsandbytes accelerate
```

### **🔑 API Configuration**

#### **Google AI Studio Setup**
1. **Visit Google AI Studio**: https://makersuite.google.com/
2. **Create Account**: Sign in with Google account
3. **Generate API Key**: Go to "Get API Key" section
4. **Copy API Key**: Save for environment setup

#### **Environment Variables**
```bash
# Primary API key (recommended)
export GEMMA_API_KEY="your_google_ai_api_key"

# Alternative environment variable names (also supported)
export GOOGLE_API_KEY="your_google_ai_api_key"

# For persistent setup, add to ~/.bashrc or ~/.zshrc
echo 'export GEMMA_API_KEY="your_api_key_here"' >> ~/.bashrc
source ~/.bashrc
```

### **🖥️ Local Model Setup**

#### **System Requirements**
- **RAM**: 8GB+ (16GB+ recommended for 4B model)
- **Storage**: 10GB+ free space for model files
- **GPU**: Optional but recommended (NVIDIA with 6GB+ VRAM)

#### **Model Configuration**
```python
# The agent will automatically download models from Hugging Face
# Supported models:
# - google/gemma-2-2b-it (2B parameters, faster)
# - google/gemma-2-9b-it (9B parameters, better quality)
# - google/gemma-7b-it (7B parameters, balanced)

# Custom model path (optional)
export GEMMA_MODEL_PATH="/path/to/local/model"
```

### **🔧 Custom Endpoint Setup**

#### **For Self-Hosted Models**
```bash
# Set custom endpoint URL
export GEMMA_ENDPOINT="http://localhost:8000/generate"

# Or configure in code:
# GemmaAgent(mode="custom", custom_endpoint="http://your-endpoint.com/api")
```

---

## 🎮 **Usage Examples**

### **🔮 Direct Chat with Gemma**
```bash
# Chat with Gemma AI
linux-ai-agent> gemma how do I secure my SSH server?

# Ask for help
linux-ai-agent> ask gemma what are the best practices for Linux security?

# Get technical assistance
linux-ai-agent> chat with gemma explain iptables firewall rules
```

### **🧠 AI-Powered Analysis**
```bash
# Analyze a topic
linux-ai-agent> ai analyze network security

# Analyze a file (combines scanning + AI analysis)
linux-ai-agent> ai analyze /var/log/auth.log

# Get comprehensive analysis
linux-ai-agent> analyze with ai suspicious process behavior
```

### **⚡ Enhanced Command Generation**
```bash
# Generate commands with AI assistance
linux-ai-agent> generate command backup database with compression

# AI-powered script generation
linux-ai-agent> generate script monitor disk usage and send alerts

# Get AI recommendations
linux-ai-agent> ai recommend improving system performance
```

### **📊 AI Statistics**
```bash
# View AI agent status including Gemma
linux-ai-agent> ai stats
```

---

## 🔧 **Configuration Options**

### **GemmaAgent Parameters**
```python
# In your code or configuration
gemma_agent = GemmaAgent(
    mode="auto",                    # auto, api, local, custom
    model_name="gemma-2-2b-it",    # Model identifier
    api_key="your_key",            # API key (optional)
    local_model_path="/path",      # Local model path (optional)
    custom_endpoint="http://...",   # Custom API endpoint (optional)
    max_tokens=1024,               # Maximum response length
    temperature=0.7                # Response creativity (0.0-1.0)
)
```

### **Environment Variables**
```bash
# API Configuration
export GEMMA_API_KEY="your_api_key"
export GOOGLE_API_KEY="your_api_key"

# Local Model Configuration
export GEMMA_MODEL_PATH="/path/to/model"
export GEMMA_MODEL_NAME="gemma-2-2b-it"

# Custom Endpoint Configuration
export GEMMA_ENDPOINT="http://localhost:8000"

# Performance Tuning
export GEMMA_MAX_TOKENS="1024"
export GEMMA_TEMPERATURE="0.7"
```

---

## 🚨 **Troubleshooting**

### **Common Issues**

#### **"Gemma AI is not available"**
```bash
# Check dependencies
pip list | grep -E "(google-generativeai|transformers|torch)"

# Verify API key
echo $GEMMA_API_KEY

# Test API connection
curl -H "Authorization: Bearer $GEMMA_API_KEY" \
     "https://generativelanguage.googleapis.com/v1/models"
```

#### **"Model loading failed"**
```bash
# Check available disk space
df -h

# Check memory usage
free -h

# Try smaller model
export GEMMA_MODEL_NAME="gemma-2-2b-it"
```

#### **"API quota exceeded"**
- Check your Google AI Studio quota
- Consider switching to local mode
- Implement request rate limiting

#### **"CUDA out of memory"**
```bash
# Use CPU mode
export CUDA_VISIBLE_DEVICES=""

# Or use model quantization
pip install bitsandbytes
```

### **Performance Optimization**

#### **For API Mode**
```bash
# Reduce response length
export GEMMA_MAX_TOKENS="512"

# Cache responses (implement in application)
# Use batch requests when possible
```

#### **For Local Mode**
```bash
# Use GPU acceleration
export CUDA_VISIBLE_DEVICES="0"

# Enable model quantization
pip install bitsandbytes accelerate

# Use smaller model variants
export GEMMA_MODEL_NAME="gemma-2-2b-it"
```

---

## 📊 **Features & Capabilities**

### **🔮 Gemma AI Features**
- **Natural Language Chat**: Direct conversation with Gemma
- **Security Analysis**: AI-powered file and system analysis
- **Command Generation**: Intelligent command creation
- **Smart Recommendations**: Context-aware suggestions
- **Multi-Modal Support**: Text analysis and generation

### **🛡️ Security Features**
- **Safety Assessment**: Automatic command safety evaluation
- **Risk Analysis**: AI-powered security risk assessment  
- **Threat Detection**: Enhanced malware analysis
- **Compliance Checking**: Policy compliance verification

### **⚙️ Integration Features**
- **Seamless Fallback**: Automatic fallback to basic features
- **Multiple Deployment Modes**: API, Local, Custom endpoints
- **Performance Monitoring**: Built-in statistics and health checks
- **Error Handling**: Robust error recovery and logging

---

## 🎯 **Best Practices**

### **🔒 Security**
- **Protect API Keys**: Never commit API keys to version control
- **Use Environment Variables**: Store sensitive configuration securely
- **Monitor Usage**: Track API usage and costs
- **Validate Responses**: Always verify AI-generated commands before execution

### **⚡ Performance**
- **Choose Appropriate Mode**: API for simplicity, Local for privacy
- **Optimize Model Size**: Use smaller models for faster responses
- **Cache Results**: Implement response caching for repeated queries
- **Monitor Resources**: Track memory and GPU usage

### **🛠️ Development**
- **Test Thoroughly**: Test all deployment modes
- **Handle Failures**: Implement proper error handling
- **Log Operations**: Enable comprehensive logging
- **Update Regularly**: Keep dependencies up to date

---

## 📚 **Additional Resources**

### **Documentation**
- [Google AI Studio](https://makersuite.google.com/)
- [Gemma Model Documentation](https://ai.google.dev/gemma)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)

### **Model Information**
- **Gemma 2B**: Fast, efficient, good for basic tasks
- **Gemma 7B**: Balanced performance and quality
- **Gemma 9B**: High quality, requires more resources

### **Community**
- [GitHub Issues](https://github.com/your-repo/issues)
- [Discord Community](https://discord.gg/your-server)
- [Documentation Wiki](https://github.com/your-repo/wiki)

---

## ✅ **Verification**

### **Test Your Setup**
```bash
# 1. Start the agent
python3 src/main.py

# 2. Check AI stats
linux-ai-agent> ai stats

# 3. Test Gemma chat
linux-ai-agent> gemma hello, are you working?

# 4. Test AI analysis
linux-ai-agent> ai analyze system security

# 5. Test command generation
linux-ai-agent> generate command list running processes
```

### **Expected Output**
- ✅ Gemma Agent: Active mode displayed
- ✅ Health Status: Healthy
- ✅ Chat responses working
- ✅ AI analysis functional
- ✅ Command generation enhanced

---

**🎉 Congratulations! Your Gemma 3 4B integration is now ready for advanced AI-powered system administration!**