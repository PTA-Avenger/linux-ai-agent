"""
Gemma 3 4B Agent for Linux AI Agent.
Provides advanced AI capabilities using Google's Gemma 3 4B model.
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from utils import get_logger, log_operation

logger = get_logger("gemma_agent")

# Optional imports for different deployment methods
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logger.warning("google-generativeai not available. API mode disabled.")

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers/torch not available. Local mode disabled.")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests not available. Custom API mode disabled.")


class GemmaAgent:
    """
    Advanced AI agent using Gemma 3 4B model.
    Supports multiple deployment modes: API, Local, and Custom endpoints.
    """
    
    def __init__(self, 
                 mode: str = "auto",
                 model_name: str = "gemma-2-2b-it",
                 api_key: Optional[str] = None,
                 local_model_path: Optional[str] = None,
                 custom_endpoint: Optional[str] = None,
                 max_tokens: int = 1024,
                 temperature: float = 0.7):
        """
        Initialize Gemma Agent with flexible deployment options.
        
        Args:
            mode: Deployment mode ('auto', 'api', 'local', 'custom')
            model_name: Model identifier (gemma-2-2b-it, gemma-2-9b-it, etc.)
            api_key: Google AI API key for API mode
            local_model_path: Path to local model files
            custom_endpoint: Custom API endpoint URL
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature (0.0-1.0)
        """
        self.mode = mode
        self.model_name = model_name
        self.api_key = api_key or os.getenv('GEMMA_API_KEY') or os.getenv('GOOGLE_API_KEY')
        self.local_model_path = local_model_path
        self.custom_endpoint = custom_endpoint
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Model instances
        self.genai_model = None
        self.local_tokenizer = None
        self.local_model = None
        
        # Statistics
        self.request_count = 0
        self.total_tokens = 0
        self.error_count = 0
        
        # Initialize based on mode
        self._initialize_model()
        
        logger.info(f"Gemma Agent initialized in {self.active_mode} mode")
    
    def _initialize_model(self):
        """Initialize the model based on available options and mode preference."""
        self.active_mode = None
        
        if self.mode == "auto":
            # Try modes in order of preference: API -> Local -> Custom
            if self._try_api_mode():
                self.active_mode = "api"
            elif self._try_local_mode():
                self.active_mode = "local"
            elif self._try_custom_mode():
                self.active_mode = "custom"
            else:
                logger.error("No available Gemma deployment method found")
                raise RuntimeError("Failed to initialize Gemma model in any mode")
        else:
            # Try specific mode
            if self.mode == "api" and self._try_api_mode():
                self.active_mode = "api"
            elif self.mode == "local" and self._try_local_mode():
                self.active_mode = "local"
            elif self.mode == "custom" and self._try_custom_mode():
                self.active_mode = "custom"
            else:
                logger.error(f"Failed to initialize Gemma in {self.mode} mode")
                raise RuntimeError(f"Gemma {self.mode} mode initialization failed")
    
    def _try_api_mode(self) -> bool:
        """Try to initialize API mode."""
        if not GENAI_AVAILABLE or not self.api_key:
            return False
        
        try:
            genai.configure(api_key=self.api_key)
            self.genai_model = genai.GenerativeModel(self.model_name)
            
            # Test with a simple query
            test_response = self.genai_model.generate_content(
                "Hello",
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=10,
                    temperature=0.1
                )
            )
            
            if test_response.text:
                logger.info("API mode initialized successfully")
                return True
            
        except Exception as e:
            logger.warning(f"API mode initialization failed: {e}")
        
        return False
    
    def _try_local_mode(self) -> bool:
        """Try to initialize local mode."""
        if not TRANSFORMERS_AVAILABLE:
            return False
        
        try:
            model_path = self.local_model_path or f"google/{self.model_name}"
            
            logger.info(f"Loading local Gemma model: {model_path}")
            
            # Load tokenizer and model
            self.local_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.local_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Test generation
            test_input = self.local_tokenizer.encode("Hello", return_tensors="pt")
            with torch.no_grad():
                test_output = self.local_model.generate(
                    test_input,
                    max_length=test_input.shape[1] + 5,
                    temperature=0.1,
                    do_sample=True
                )
            
            if test_output is not None:
                logger.info("Local mode initialized successfully")
                return True
            
        except Exception as e:
            logger.warning(f"Local mode initialization failed: {e}")
        
        return False
    
    def _try_custom_mode(self) -> bool:
        """Try to initialize custom endpoint mode."""
        if not REQUESTS_AVAILABLE or not self.custom_endpoint:
            return False
        
        try:
            # Test custom endpoint
            response = requests.post(
                self.custom_endpoint,
                json={
                    "prompt": "Hello",
                    "max_tokens": 10,
                    "temperature": 0.1
                },
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("Custom endpoint mode initialized successfully")
                return True
            
        except Exception as e:
            logger.warning(f"Custom endpoint mode initialization failed: {e}")
        
        return False
    
    def generate_response(self, 
                         prompt: str, 
                         context: Optional[str] = None,
                         system_prompt: Optional[str] = None,
                         max_tokens: Optional[int] = None,
                         temperature: Optional[float] = None) -> Dict[str, Any]:
        """
        Generate a response using the active Gemma model.
        
        Args:
            prompt: User prompt/question
            context: Additional context information
            system_prompt: System instruction
            max_tokens: Override default max tokens
            temperature: Override default temperature
            
        Returns:
            Dictionary with response and metadata
        """
        start_time = time.time()
        
        try:
            # Build full prompt
            full_prompt = self._build_prompt(prompt, context, system_prompt)
            
            # Use provided parameters or defaults
            max_tokens = max_tokens or self.max_tokens
            temperature = temperature or self.temperature
            
            # Generate based on active mode
            if self.active_mode == "api":
                response = self._generate_api(full_prompt, max_tokens, temperature)
            elif self.active_mode == "local":
                response = self._generate_local(full_prompt, max_tokens, temperature)
            elif self.active_mode == "custom":
                response = self._generate_custom(full_prompt, max_tokens, temperature)
            else:
                raise RuntimeError("No active generation mode")
            
            # Update statistics
            self.request_count += 1
            processing_time = time.time() - start_time
            
            result = {
                "success": True,
                "response": response,
                "mode": self.active_mode,
                "model": self.model_name,
                "processing_time": f"{processing_time:.2f}s",
                "tokens_generated": len(response.split()) if response else 0,
                "prompt_length": len(full_prompt)
            }
            
            log_operation(logger, "GENERATE_RESPONSE", {
                "mode": self.active_mode,
                "prompt_length": len(prompt),
                "response_length": len(response) if response else 0,
                "processing_time": processing_time
            })
            
            return result
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Response generation failed: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "mode": self.active_mode,
                "processing_time": f"{time.time() - start_time:.2f}s"
            }
    
    def _build_prompt(self, prompt: str, context: Optional[str], system_prompt: Optional[str]) -> str:
        """Build the complete prompt with context and system instructions."""
        parts = []
        
        if system_prompt:
            parts.append(f"System: {system_prompt}")
        
        if context:
            parts.append(f"Context: {context}")
        
        parts.append(f"User: {prompt}")
        parts.append("Assistant:")
        
        return "\n\n".join(parts)
    
    def _generate_api(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate response using Google AI API."""
        try:
            response = self.genai_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature
                )
            )
            
            return response.text if response.text else ""
            
        except Exception as e:
            logger.error(f"API generation failed: {e}")
            raise
    
    def _generate_local(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate response using local model."""
        try:
            # Tokenize input
            inputs = self.local_tokenizer.encode(prompt, return_tensors="pt")
            
            # Generate
            with torch.no_grad():
                outputs = self.local_model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.local_tokenizer.eos_token_id
                )
            
            # Decode response (exclude input tokens)
            response_tokens = outputs[0][inputs.shape[1]:]
            response = self.local_tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Local generation failed: {e}")
            raise
    
    def _generate_custom(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate response using custom endpoint."""
        try:
            response = requests.post(
                self.custom_endpoint,
                json={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                },
                timeout=30
            )
            
            response.raise_for_status()
            data = response.json()
            
            return data.get("response", data.get("text", ""))
            
        except Exception as e:
            logger.error(f"Custom endpoint generation failed: {e}")
            raise
    
    def analyze_security_context(self, 
                                scan_results: Dict[str, Any], 
                                file_path: str) -> Dict[str, Any]:
        """
        Analyze security scan results using Gemma's advanced reasoning.
        
        Args:
            scan_results: Results from security scans
            file_path: Path of the scanned file
            
        Returns:
            Enhanced security analysis
        """
        context = json.dumps(scan_results, indent=2)
        
        system_prompt = """You are a cybersecurity expert analyzing file scan results. 
        Provide clear, actionable security analysis focusing on:
        1. Risk assessment and severity
        2. Potential attack vectors
        3. Recommended actions
        4. False positive likelihood
        Keep responses concise and practical."""
        
        prompt = f"""Analyze these security scan results for file '{file_path}':

{context}

Provide a comprehensive security assessment including risk level, potential threats, and recommended actions."""
        
        result = self.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=512,
            temperature=0.3
        )
        
        if result["success"]:
            return {
                "analysis": result["response"],
                "ai_risk_score": self._extract_risk_score(result["response"]),
                "recommendations": self._extract_recommendations(result["response"]),
                "confidence": 0.8,  # Gemma provides high-quality analysis
                "model_used": f"Gemma {self.model_name}"
            }
        else:
            return {
                "analysis": "AI analysis failed",
                "error": result.get("error"),
                "ai_risk_score": 0,
                "recommendations": [],
                "confidence": 0.0
            }
    
    def generate_system_command(self, description: str) -> Dict[str, Any]:
        """
        Generate system commands from natural language using Gemma.
        
        Args:
            description: Natural language description of desired command
            
        Returns:
            Generated command with safety assessment
        """
        system_prompt = """You are a Linux system administration expert. 
        Generate safe, accurate shell commands from natural language descriptions.
        Always include safety warnings for potentially dangerous operations.
        Format: Provide the command, explanation, and safety level (safe/moderate/dangerous)."""
        
        prompt = f"""Generate a Linux shell command for: {description}

Requirements:
- Provide the exact command
- Explain what it does
- Assess safety level
- Include any necessary warnings"""
        
        result = self.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=256,
            temperature=0.2
        )
        
        if result["success"]:
            response = result["response"]
            return {
                "success": True,
                "command": self._extract_command(response),
                "explanation": response,
                "safety_level": self._assess_command_safety(response),
                "confidence": 0.9,
                "model_used": f"Gemma {self.model_name}"
            }
        else:
            return {
                "success": False,
                "error": result.get("error"),
                "command": None,
                "explanation": "Command generation failed"
            }
    
    def provide_recommendations(self, context: str) -> List[Dict[str, Any]]:
        """
        Provide intelligent recommendations based on context.
        
        Args:
            context: Context information for recommendations
            
        Returns:
            List of recommendations with priorities
        """
        system_prompt = """You are an AI assistant for system administration and security.
        Provide practical, prioritized recommendations based on the given context.
        Format each recommendation with priority (HIGH/MEDIUM/LOW) and rationale."""
        
        prompt = f"""Based on this context, provide 3-5 actionable recommendations:

{context}

Focus on security, efficiency, and best practices."""
        
        result = self.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=384,
            temperature=0.4
        )
        
        if result["success"]:
            return self._parse_recommendations(result["response"])
        else:
            return [{
                "priority": "LOW",
                "action": "manual_review",
                "description": "AI recommendations unavailable - manual review required",
                "rationale": result.get("error", "Unknown error")
            }]
    
    def _extract_risk_score(self, analysis: str) -> int:
        """Extract risk score from analysis text."""
        # Simple extraction logic - could be enhanced
        analysis_lower = analysis.lower()
        if any(word in analysis_lower for word in ['critical', 'severe', 'high risk']):
            return 85
        elif any(word in analysis_lower for word in ['moderate', 'medium risk']):
            return 50
        elif any(word in analysis_lower for word in ['low risk', 'minimal']):
            return 20
        else:
            return 30  # Default moderate risk
    
    def _extract_recommendations(self, analysis: str) -> List[str]:
        """Extract recommendations from analysis text."""
        recommendations = []
        lines = analysis.split('\n')
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['recommend', 'should', 'action', 'step']):
                if line and not line.startswith('#'):
                    recommendations.append(line)
        
        return recommendations[:5]  # Limit to top 5
    
    def _extract_command(self, response: str) -> Optional[str]:
        """Extract command from Gemma response."""
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for lines that look like commands
            if line.startswith('$') or line.startswith('sudo') or line.startswith('#'):
                return line.lstrip('$# ')
            elif '`' in line:
                # Extract from code blocks
                start = line.find('`')
                end = line.rfind('`')
                if start != end and start != -1:
                    return line[start+1:end]
        
        return None
    
    def _assess_command_safety(self, response: str) -> str:
        """Assess command safety from response."""
        response_lower = response.lower()
        
        if any(word in response_lower for word in ['dangerous', 'warning', 'careful', 'destructive']):
            return "dangerous"
        elif any(word in response_lower for word in ['moderate', 'caution', 'sudo']):
            return "moderate"
        else:
            return "safe"
    
    def _parse_recommendations(self, response: str) -> List[Dict[str, Any]]:
        """Parse recommendations from Gemma response."""
        recommendations = []
        lines = response.split('\n')
        
        current_rec = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for priority indicators
            if any(priority in line.upper() for priority in ['HIGH', 'MEDIUM', 'LOW']):
                if current_rec:
                    recommendations.append(current_rec)
                
                priority = 'MEDIUM'  # default
                if 'HIGH' in line.upper():
                    priority = 'HIGH'
                elif 'LOW' in line.upper():
                    priority = 'LOW'
                
                current_rec = {
                    "priority": priority,
                    "action": "review_and_implement",
                    "description": line,
                    "rationale": "AI-generated recommendation"
                }
            elif current_rec:
                # Add to current recommendation description
                current_rec["description"] += " " + line
        
        if current_rec:
            recommendations.append(current_rec)
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get Gemma agent statistics."""
        return {
            "active_mode": self.active_mode,
            "model_name": self.model_name,
            "request_count": self.request_count,
            "total_tokens": self.total_tokens,
            "error_count": self.error_count,
            "success_rate": (self.request_count - self.error_count) / max(self.request_count, 1),
            "api_available": GENAI_AVAILABLE and bool(self.api_key),
            "local_available": TRANSFORMERS_AVAILABLE,
            "custom_available": REQUESTS_AVAILABLE and bool(self.custom_endpoint)
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the Gemma agent."""
        try:
            test_result = self.generate_response(
                prompt="Hello, are you working?",
                max_tokens=20,
                temperature=0.1
            )
            
            return {
                "healthy": test_result["success"],
                "mode": self.active_mode,
                "response_time": test_result.get("processing_time", "unknown"),
                "error": test_result.get("error") if not test_result["success"] else None
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "mode": self.active_mode
            }