"""
Enhanced Intent Parser using modern NLP techniques.
This replaces the regex-based approach with semantic understanding.
"""

import os
import sys
import json
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pickle
import re

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from utils import get_logger, log_operation

# Import AI/ML libraries
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    import torch
    ENHANCED_NLP_AVAILABLE = True
except ImportError:
    print("Enhanced NLP libraries not available. Install: pip install sentence-transformers torch scikit-learn")
    ENHANCED_NLP_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

logger = get_logger("enhanced_intent_parser")


class EnhancedIntentParser:
    """
    Advanced intent parser using semantic embeddings and machine learning.
    Now includes Unix-style flag parsing and improved command recognition.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = "ai_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.model_name = model_name
        self.embedding_model = None
        self.nlp = None
        
        # Enhanced training data with better coverage
        self.training_data = {
            "create_file": [
                "create a new file called test.txt",
                "make a file named document.pdf", 
                "generate file output.json",
                "touch new_file.py",
                "I need to create a configuration file",
                "please make a new text file for me"
            ],
            "read_file": [
                "read the contents of config.txt",
                "show me what's in the log file",
                "display the content of readme.md",
                "cat the file data.json",
                "I want to see what's inside document.txt",
                "open and read the file settings.ini"
            ],
            "scan_file": [
                "scan this file for malware",
                "check if download.exe is infected",
                "analyze file.zip for viruses",
                "run security scan on document.pdf",
                "is this file safe to open?",
                "perform malware detection on suspicious.exe"
            ],
            "heuristic_scan": [
                "heuristic scan /path/to/file",
                "run heuristic analysis on file.exe",
                "perform entropy analysis on document.pdf",
                "heuristic check suspicious.zip",
                "analyze file using heuristics",
                "entropy scan the downloaded file"
            ],
            "scan_directory": [
                "scan the entire downloads folder",
                "check all files in /tmp for viruses",
                "analyze the directory recursively",
                "run full scan on project folder",
                "scan this folder and all subfolders",
                "recursive scan /home/user/downloads"
            ],
            "list_directory": [
                "list files in directory",
                "ls /home/user",
                "show directory contents",
                "list all files",
                "ls -l /var/log",
                "dir listing with details"
            ],
            "quarantine_file": [
                "quarantine the infected file",
                "isolate suspicious.exe",
                "move the malware to quarantine",
                "put this dangerous file in isolation",
                "quarantine the threat"
            ],
            "system_stats": [
                "show system information",
                "what's the current CPU usage?",
                "display memory statistics",
                "how is the system performing?",
                "system status report",
                "check system health"
            ],
            "disk_usage": [
                "check disk space",
                "how much storage is available?",
                "show disk usage statistics",
                "storage information please",
                "disk space report",
                "df -h"
            ],
            "help": [
                "what can you do?",
                "show available commands",
                "help me with options",
                "what are your capabilities?",
                "list all functions",
                "help",
                "--help"
            ],
            "generate_command": [
                "generate command to backup files",
                "create a command that monitors logs",
                "make script to clean up system",
                "generate backup command for /home/user",
                "create command to find large files",
                "write script that scans for malware"
            ],
            "generate_script": [
                "generate script that backs up /var/log hourly",
                "create maintenance script for the system",
                "make script to automate file cleanup",
                "write script for daily security scans",
                "generate automated backup script"
            ]
        }
        
        # Unix-style flag patterns
        self.flag_patterns = {
            "-l": "long_format",
            "--long": "long_format",
            "-a": "all_files",
            "--all": "all_files",
            "-r": "recursive",
            "--recursive": "recursive",
            "-f": "force",
            "--force": "force",
            "-v": "verbose",
            "--verbose": "verbose",
            "-q": "quiet",
            "--quiet": "quiet",
            "-h": "help",
            "--help": "help",
            "--version": "version"
        }
        
        # Initialize models
        self._initialize_models()
        
        # Create intent embeddings
        self.intent_embeddings = None
        self.intent_labels = None
        self._create_intent_embeddings()
        
        logger.info(f"Enhanced Intent Parser initialized with model: {model_name}")
    
    def _initialize_models(self):
        """Initialize NLP models."""
        if not ENHANCED_NLP_AVAILABLE:
            logger.warning("Enhanced NLP not available, falling back to basic implementation")
            return
        
        try:
            # Load sentence transformer model
            model_cache = self.cache_dir / f"sentence_transformer_{self.model_name.replace('/', '_')}"
            if model_cache.exists():
                self.embedding_model = SentenceTransformer(str(model_cache))
            else:
                self.embedding_model = SentenceTransformer(self.model_name)
                self.embedding_model.save(str(model_cache))
            
            logger.info("Sentence transformer model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading sentence transformer: {e}")
            ENHANCED_NLP_AVAILABLE = False
        
        # Try to load spaCy model
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy model loaded successfully")
            except OSError:
                logger.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
                SPACY_AVAILABLE = False
    
    def _create_intent_embeddings(self):
        """Create embeddings for all training examples."""
        if not ENHANCED_NLP_AVAILABLE or self.embedding_model is None:
            return
        
        embeddings_cache = self.cache_dir / "intent_embeddings.pkl"
        
        if embeddings_cache.exists():
            try:
                with open(embeddings_cache, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.intent_embeddings = cached_data['embeddings']
                    self.intent_labels = cached_data['labels']
                logger.info("Loaded cached intent embeddings")
                return
            except Exception as e:
                logger.warning(f"Failed to load cached embeddings: {e}")
        
        # Create embeddings for training data
        all_examples = []
        all_labels = []
        
        for intent, examples in self.training_data.items():
            for example in examples:
                all_examples.append(example)
                all_labels.append(intent)
        
        try:
            self.intent_embeddings = self.embedding_model.encode(all_examples)
            self.intent_labels = all_labels
            
            # Cache the embeddings
            cache_data = {
                'embeddings': self.intent_embeddings,
                'labels': self.intent_labels
            }
            with open(embeddings_cache, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.info(f"Created embeddings for {len(all_examples)} training examples")
            
        except Exception as e:
            logger.error(f"Error creating intent embeddings: {e}")
    
    def parse_flags(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Parse Unix-style flags from command text.
        
        Args:
            text: Input command text
            
        Returns:
            Tuple of (cleaned_text, flags_dict)
        """
        flags = {}
        cleaned_text = text
        
        # Find all flag patterns
        for flag, flag_name in self.flag_patterns.items():
            # Look for the flag as a separate word
            flag_pattern = r'\b' + re.escape(flag) + r'\b'
            if re.search(flag_pattern, text):
                flags[flag_name] = True
                # Remove the flag from text
                cleaned_text = re.sub(flag_pattern, '', cleaned_text)
        
        # Clean up extra whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        return cleaned_text, flags
    
    def parse_intent(self, text: str) -> Dict[str, Any]:
        """
        Parse intent using semantic similarity with improved flag and command support.
        
        Args:
            text: User input text
        
        Returns:
            Dictionary with parsed intent information
        """
        if not text or not text.strip():
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "parameters": {},
                "flags": {},
                "original_text": text,
                "method": "fallback"
            }
        
        original_text = text.strip()
        
        # Parse flags first
        cleaned_text, flags = self.parse_flags(original_text)
        
        # Handle special flag cases
        if flags.get("help"):
            return {
                "intent": "help",
                "confidence": 1.0,
                "parameters": {},
                "flags": flags,
                "original_text": original_text,
                "method": "flag_detection"
            }
        
        # Try enhanced NLP first
        if ENHANCED_NLP_AVAILABLE and self.embedding_model is not None:
            result = self._parse_with_embeddings(cleaned_text)
            if result["confidence"] > 0.3:  # Threshold for semantic matching
                result["flags"] = flags
                result["original_text"] = original_text
                return result
        
        # Fallback to enhanced regex matching
        result = self._parse_with_enhanced_regex(cleaned_text)
        result["flags"] = flags
        result["original_text"] = original_text
        return result
    
    def _parse_with_embeddings(self, text: str) -> Dict[str, Any]:
        """Parse intent using semantic embeddings."""
        try:
            # Get embedding for input text
            query_embedding = self.embedding_model.encode([text])
            
            # Calculate similarities with all training examples
            similarities = cosine_similarity(query_embedding, self.intent_embeddings)[0]
            
            # Find best match
            if NUMPY_AVAILABLE:
                best_idx = np.argmax(similarities)
                best_confidence = float(similarities[best_idx])
                # Get top 3 similar intents for context
                top_indices = np.argsort(similarities)[-3:][::-1]
            else:
                # Fallback without numpy
                best_idx = similarities.index(max(similarities))
                best_confidence = float(similarities[best_idx])
                # Simple sorting fallback
                indexed_sims = [(i, sim) for i, sim in enumerate(similarities)]
                indexed_sims.sort(key=lambda x: x[1], reverse=True)
                top_indices = [x[0] for x in indexed_sims[:3]]
                
            best_intent = self.intent_labels[best_idx]
            
            # Extract parameters using NLP
            parameters = self._extract_parameters_nlp(text, best_intent)
            similar_intents = [
                {
                    "intent": self.intent_labels[idx],
                    "confidence": float(similarities[idx]),
                    "example": self._get_example_for_index(idx)
                }
                for idx in top_indices
            ]
            
            result = {
                "intent": best_intent,
                "confidence": best_confidence,
                "parameters": parameters,
                "original_text": text,
                "method": "semantic_embedding",
                "similar_intents": similar_intents,
                "embedding_model": self.model_name
            }
            
            log_operation(logger, "PARSE_INTENT_SEMANTIC", {
                "text": text,
                "intent": best_intent,
                "confidence": best_confidence,
                "method": "embeddings"
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in semantic parsing: {e}")
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "parameters": {},
                "original_text": text,
                "method": "error",
                "error": str(e)
            }
    
    def _extract_parameters_nlp(self, text: str, intent: str) -> Dict[str, Any]:
        """Extract parameters using NLP techniques."""
        parameters = {}
        
        # Use spaCy for named entity recognition if available
        if SPACY_AVAILABLE and self.nlp:
            try:
                doc = self.nlp(text)
                
                # Extract file paths and names
                for token in doc:
                    if token.like_url or token.like_email:
                        continue
                        
                    # Look for file-like patterns
                    if ('.' in token.text and 
                        len(token.text) > 3 and 
                        not token.is_punct and 
                        not token.is_space):
                        
                        # Check if it looks like a filename
                        extensions = ['.txt', '.py', '.json', '.xml', '.log', '.exe', '.pdf', '.doc', '.zip', '.tar', '.gz']
                        if any(ext in token.text.lower() for ext in extensions):
                            parameters["path"] = token.text
                            parameters["is_file"] = True
                            break
                
                # Extract entities
                for ent in doc.ents:
                    if ent.label_ in ["FILE", "ORG", "PRODUCT"]:
                        if "path" not in parameters:
                            parameters["path"] = ent.text
                
                # Extract directory-like paths
                for token in doc:
                    if (token.text.startswith('/') or 
                        token.text.startswith('./') or 
                        token.text.startswith('../')):
                        parameters["path"] = token.text
                        parameters["is_directory"] = True
                        break
                        
            except Exception as e:
                logger.warning(f"Error in NLP parameter extraction: {e}")
        
        # Enhanced fallback: pattern matching with better coverage
        if "path" not in parameters:
            import re
            
            # Look for quoted strings (likely filenames)
            quoted_match = re.search(r'["\']([^"\']+)["\']', text)
            if quoted_match:
                parameters["path"] = quoted_match.group(1)
            else:
                # Look for Unix-style paths
                unix_path_match = re.search(r'(/[^\s]+)', text)
                if unix_path_match:
                    parameters["path"] = unix_path_match.group(1)
                    parameters["is_directory"] = True
                else:
                    # Look for file extensions
                    file_match = re.search(r'(\S+\.\w+)', text)
                    if file_match:
                        parameters["path"] = file_match.group(1)
                        parameters["is_file"] = True
        
        # Intent-specific parameter extraction
        if intent in ["scan_file", "read_file", "create_file", "delete_file", "heuristic_scan"]:
            if "path" not in parameters:
                # Try to extract the last word that might be a filename
                words = text.split()
                for word in reversed(words):
                    if ('.' in word and len(word) > 3) or word.startswith('/'):
                        parameters["path"] = word
                        break
        
        return parameters
    
    def _parse_with_enhanced_regex(self, text: str) -> Dict[str, Any]:
        """Enhanced regex-based parsing with better pattern coverage."""
        import re
        
        # Enhanced patterns with better coverage
        patterns = {
            "create_file": [
                r"(?:create|make|touch|generate).*?(?:file|document)",
                r"new.*?file"
            ],
            "read_file": [
                r"(?:read|show|display|cat|open).*?(?:file|content)",
                r"(?:cat|less|more)\s+\S+"
            ],
            "scan_file": [
                r"(?:scan|check|analyze).*?(?:file|malware|virus)",
                r"(?:malware|virus).*?(?:scan|check)"
            ],
            "heuristic_scan": [
                r"heuristic.*?(?:scan|analysis|check)",
                r"entropy.*?(?:scan|analysis)",
                r"(?:scan|analyze).*?heuristic"
            ],
            "scan_directory": [
                r"(?:scan|check).*?(?:directory|folder|dir)",
                r"recursive.*?scan"
            ],
            "list_directory": [
                r"(?:list|ls|dir)(?:\s|$)",
                r"show.*?(?:directory|folder|files)",
                r"ls\s+\S+"
            ],
            "system_stats": [
                r"(?:system|cpu|memory|performance|stats|status)",
                r"(?:top|htop|ps)"
            ],
            "disk_usage": [
                r"(?:disk|storage|space|usage)",
                r"(?:df|du)(?:\s|$)"
            ],
            "help": [
                r"(?:help|commands|what.*do|capabilities)",
                r"--help|-h"
            ]
        }
        
        best_match = {"intent": "unknown", "confidence": 0.0}
        
        for intent, intent_patterns in patterns.items():
            for pattern in intent_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    # Calculate confidence based on match quality
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        match_ratio = len(match.group(0)) / len(text)
                        confidence = min(0.8, 0.5 + match_ratio * 0.3)  # Max 0.8 for regex
                        
                        if confidence > best_match["confidence"]:
                            best_match = {"intent": intent, "confidence": confidence}
        
        return {
            "intent": best_match["intent"],
            "confidence": best_match["confidence"],
            "parameters": self._extract_parameters_enhanced(text, best_match["intent"]),
            "method": "enhanced_regex"
        }
    
    def _extract_parameters_enhanced(self, text: str, intent: str) -> Dict[str, Any]:
        """Enhanced parameter extraction using NLP and regex."""
        parameters = {}
        
        # Use spaCy for named entity recognition if available
        if SPACY_AVAILABLE and self.nlp:
            try:
                doc = self.nlp(text)
                
                # Extract file paths and names
                for token in doc:
                    if token.like_url or token.like_email:
                        continue
                        
                    # Look for file-like patterns
                    if ('.' in token.text and 
                        len(token.text) > 3 and 
                        not token.is_punct and 
                        not token.is_space):
                        
                        # Check if it looks like a filename
                        extensions = ['.txt', '.py', '.json', '.xml', '.log', '.exe', '.pdf', '.doc']
                        if any(ext in token.text.lower() for ext in extensions):
                            parameters["path"] = token.text
                            parameters["is_file"] = True
                            break
                
                # Extract entities
                for ent in doc.ents:
                    if ent.label_ in ["FILE", "ORG", "PRODUCT"]:
                        if "path" not in parameters:
                            parameters["path"] = ent.text
                
                # Extract directory-like paths
                for token in doc:
                    if (token.text.startswith('/') or 
                        token.text.startswith('./') or 
                        token.text.startswith('../')):
                        parameters["path"] = token.text
                        parameters["is_directory"] = True
                        break
                        
            except Exception as e:
                logger.warning(f"Error in enhanced NLP parameter extraction: {e}")
        
        # Fallback: simple pattern matching
        if "path" not in parameters:
            import re
            
            # Look for quoted strings (likely filenames)
            quoted_match = re.search(r'["\']([^"\']+)["\']', text)
            if quoted_match:
                parameters["path"] = quoted_match.group(1)
            else:
                # Look for file extensions
                file_match = re.search(r'(\S+\.\w+)', text)
                if file_match:
                    parameters["path"] = file_match.group(1)
                    parameters["is_file"] = True
        
        # Intent-specific parameter extraction
        if intent in ["scan_file", "read_file", "create_file", "delete_file"]:
            if "path" not in parameters:
                # Try to extract the last word that might be a filename
                words = text.split()
                for word in reversed(words):
                    if '.' in word and len(word) > 3:
                        parameters["path"] = word
                        break
        
        return parameters
    
    def _get_example_for_index(self, idx: int) -> str:
        """Get the training example for a given index."""
        current_idx = 0
        for intent, examples in self.training_data.items():
            if current_idx <= idx < current_idx + len(examples):
                return examples[idx - current_idx]
            current_idx += len(examples)
        return ""
    
    def add_training_example(self, intent: str, example: str):
        """Add a new training example and update embeddings."""
        if intent not in self.training_data:
            self.training_data[intent] = []
        
        self.training_data[intent].append(example)
        
        # Recreate embeddings
        self._create_intent_embeddings()
        
        logger.info(f"Added training example for intent '{intent}': {example}")
    
    def get_intent_suggestions(self, partial_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get intent suggestions based on partial input."""
        if not partial_text or not ENHANCED_NLP_AVAILABLE:
            return []
        
        try:
            # Get embedding for partial text
            query_embedding = self.embedding_model.encode([partial_text])
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.intent_embeddings)[0]
            
            # Get top suggestions
            if NUMPY_AVAILABLE:
                top_indices = np.argsort(similarities)[-limit:][::-1]
            else:
                # Fallback without numpy
                indexed_sims = [(i, sim) for i, sim in enumerate(similarities)]
                indexed_sims.sort(key=lambda x: x[1], reverse=True)
                top_indices = [x[0] for x in indexed_sims[:limit]]
            
            suggestions = []
            seen_intents = set()
            
            for idx in top_indices:
                intent = self.intent_labels[idx]
                if intent not in seen_intents:
                    suggestions.append({
                        "intent": intent,
                        "confidence": float(similarities[idx]),
                        "example": self._get_example_for_index(idx),
                        "description": self._get_intent_description(intent)
                    })
                    seen_intents.add(intent)
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error getting suggestions: {e}")
            return []
    
    def _get_intent_description(self, intent: str) -> str:
        """Get description for an intent."""
        descriptions = {
            "create_file": "Create a new file",
            "read_file": "Read contents of a file", 
            "scan_file": "Scan a file for malware",
            "scan_directory": "Scan a directory for malware",
            "system_stats": "Show system statistics",
            "disk_usage": "Check disk usage",
            "help": "Show help information"
        }
        return descriptions.get(intent, f"Execute {intent} operation")
    
    def analyze_conversation_context(self, conversation_history: List[str]) -> Dict[str, Any]:
        """Analyze conversation context to improve intent understanding."""
        if not conversation_history or not ENHANCED_NLP_AVAILABLE:
            return {}
        
        try:
            # Get embeddings for conversation history
            history_embeddings = self.embedding_model.encode(conversation_history)
            
            # Cluster conversation topics
            if len(conversation_history) > 3:
                kmeans = KMeans(n_clusters=min(3, len(conversation_history)//2))
                clusters = kmeans.fit_predict(history_embeddings)
                
                # Analyze dominant topics
                dominant_cluster = max(set(clusters), key=list(clusters).count)
                dominant_messages = [
                    conversation_history[i] for i, c in enumerate(clusters) 
                    if c == dominant_cluster
                ]
            else:
                dominant_messages = conversation_history
            
            # Extract context features
            context = {
                "conversation_length": len(conversation_history),
                "dominant_topic_messages": dominant_messages,
                "recent_intents": self._extract_recent_intents(conversation_history[-3:]),
                "user_expertise_level": self._estimate_expertise_level(conversation_history)
            }
            
            return context
            
        except Exception as e:
            logger.error(f"Error analyzing conversation context: {e}")
            return {}
    
    def _extract_recent_intents(self, recent_messages: List[str]) -> List[str]:
        """Extract intents from recent messages."""
        intents = []
        for message in recent_messages:
            result = self.parse_intent(message)
            if result["confidence"] > 0.5:
                intents.append(result["intent"])
        return intents
    
    def _estimate_expertise_level(self, conversation_history: List[str]) -> str:
        """Estimate user expertise level based on conversation."""
        technical_terms = [
            "malware", "entropy", "quarantine", "hash", "signature",
            "recursive", "permissions", "daemon", "process", "thread"
        ]
        
        technical_count = sum(
            1 for message in conversation_history 
            for term in technical_terms 
            if term in message.lower()
        )
        
        if technical_count > len(conversation_history) * 0.3:
            return "expert"
        elif technical_count > len(conversation_history) * 0.1:
            return "intermediate"
        else:
            return "beginner"