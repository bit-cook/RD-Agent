"""
LLM API with retry mechanism
Based on RD-Agent's retry logic
"""

import os
import re
import time
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from litellm import completion, supports_response_schema
from litellm.exceptions import (
    RateLimitError, 
    Timeout,
    BadRequestError,
)


# 全局日志目录
LOG_DIR: Optional[Path] = None
CALL_COUNT = 0


def set_log_dir(log_dir: str):
    """设置 LLM 交互日志目录"""
    global LOG_DIR
    LOG_DIR = Path(log_dir)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[LLM] 日志目录: {LOG_DIR}")


class LLMClient:
    """LLM client with retry mechanism (based on RD-Agent)"""
    
    def __init__(
        self,
        model: str = None,
        max_retry: int = 10,
        retry_wait: int = 1,
        temperature: float = 0.7,
    ):
        # Use env var or default
        self.model = model or os.getenv("LITELLM_CHAT_MODEL", "gpt-4")
        self.max_retry = max_retry
        self.retry_wait = retry_wait
        self.temperature = temperature
        
    def chat(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
    ) -> str:
        """
        Call LLM with retry
        
        Args:
            messages: Chat messages
            json_mode: If True, expect JSON response
            
        Returns:
            LLM response content
        """
        # Copy messages to avoid modifying original
        messages = [m.copy() for m in messages]
        
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        
        # Check if model supports response_format (from RD-Agent)
        use_response_format = False
        if json_mode:
            try:
                if supports_response_schema(model=self.model):
                    kwargs["response_format"] = {"type": "json_object"}
                    use_response_format = True
                else:
                    # Model doesn't support response_format, add hint to prompt
                    self._add_json_hint(messages)
                    kwargs["messages"] = messages
            except:
                # If check fails, just add hint to prompt
                self._add_json_hint(messages)
                kwargs["messages"] = messages
        
        for i in range(self.max_retry):
            try:
                response = completion(**kwargs)
                content = response.choices[0].message.content
                
                # Parse JSON if needed
                if json_mode:
                    content = self._parse_json(content)
                
                # 保存交互日志
                self._save_interaction(messages, content)
                    
                return content
                
            except RateLimitError as e:
                wait = self._parse_wait_time(str(e))
                print(f"[LLM] Rate limited, waiting {wait}s... ({i+1}/{self.max_retry})")
                time.sleep(wait)
                
            except Timeout:
                print(f"[LLM] Timeout, retrying... ({i+1}/{self.max_retry})")
                time.sleep(self.retry_wait)
                
            except BadRequestError as e:
                error_str = str(e).lower()
                # Content policy violation - don't retry
                if "content management policy" in error_str:
                    print(f"[LLM] Content policy violation: {e}")
                    raise
                # response_format not supported - retry without it
                if "response_format" in error_str or "unsupportedparams" in error_str:
                    print(f"[LLM] response_format not supported, retrying without it...")
                    if "response_format" in kwargs:
                        del kwargs["response_format"]
                        self._add_json_hint(messages)
                        kwargs["messages"] = messages
                    continue
                print(f"[LLM] Bad request: {e}, retrying... ({i+1}/{self.max_retry})")
                time.sleep(self.retry_wait)
                
            except json.JSONDecodeError:
                print(f"[LLM] Invalid JSON response, retrying... ({i+1}/{self.max_retry})")
                time.sleep(self.retry_wait)
                
            except Exception as e:
                print(f"[LLM] Error: {e}, retrying... ({i+1}/{self.max_retry})")
                time.sleep(self.retry_wait)
        
        raise RuntimeError(f"Failed after {self.max_retry} retries")
    
    def _save_interaction(self, messages: List[Dict], response: str, error: str = None):
        """保存 LLM 交互日志到文件"""
        global LOG_DIR, CALL_COUNT
        if LOG_DIR is None:
            return
        
        CALL_COUNT += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOG_DIR / f"llm_{CALL_COUNT:03d}_{timestamp}.json"
        
        log_data = {
            "call_id": CALL_COUNT,
            "timestamp": timestamp,
            "model": self.model,
            "messages": messages,
            "response": response,
            "error": error,
        }
        
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        print(f"[LLM] 交互日志已保存: {log_file.name}")
    
    def _add_json_hint(self, messages: List[Dict]) -> None:
        """Add JSON format hint to last message (from RD-Agent)"""
        if messages:
            last_msg = messages[-1]
            if "Please respond in JSON format" not in last_msg.get("content", ""):
                last_msg["content"] = last_msg.get("content", "") + "\n\nPlease respond in JSON format."
    
    def _parse_json(self, content: str) -> str:
        """Parse JSON from response, handling code blocks (from RD-Agent)"""
        # Try direct parse
        try:
            json.loads(content)
            return content
        except json.JSONDecodeError:
            pass
        
        # Try extract from ```json ... ``` code block
        match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            try:
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                pass
        
        # Try extract from ``` ... ``` code block
        match = re.search(r"```\s*(.*?)\s*```", content, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            try:
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                pass
        
        # Return original and let caller handle error
        return content
    
    def _parse_wait_time(self, error_msg: str) -> int:
        """Parse wait time from rate limit error (from RD-Agent)"""
        match = re.search(r"retry after (\d+) seconds", error_msg, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return self.retry_wait


# Default client
_default_client: Optional[LLMClient] = None


def get_client() -> LLMClient:
    """Get default LLM client"""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


def chat(messages: List[Dict[str, str]], json_mode: bool = False) -> str:
    """Convenience function for LLM chat"""
    return get_client().chat(messages, json_mode)
