"""
Tools available for the Agent
"""

import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional


class ToolResult:
    """Result from tool execution"""
    
    def __init__(self, success: bool, output: str, error: str = ""):
        self.success = success
        self.output = output
        self.error = error
        
    def __str__(self):
        if self.success:
            return f"[OK] {self.output}"
        return f"[ERROR] {self.error}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
        }


class AgentTools:
    """Tools for Agent to interact with the environment"""
    
    def __init__(self, workspace: str):
        """
        Args:
            workspace: Working directory for the agent
        """
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        
    def run_code(self, code: str, timeout: int = 300) -> ToolResult:
        """
        Execute Python code
        
        Args:
            code: Python code to execute
            timeout: Max execution time in seconds
            
        Returns:
            ToolResult with stdout/stderr
        """
        # Write code to temp file
        code_file = self.workspace / "temp_code.py"
        code_file.write_text(code)
        
        try:
            result = subprocess.run(
                [sys.executable, str(code_file)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.workspace),
            )
            
            output = result.stdout
            if result.returncode != 0:
                return ToolResult(
                    success=False,
                    output=output,
                    error=result.stderr,
                )
            return ToolResult(success=True, output=output)
            
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error=f"Code execution timeout ({timeout}s)",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=str(e),
            )
    
    def read_file(self, path: str, max_lines: int = 500) -> ToolResult:
        """
        Read a file
        
        Args:
            path: File path (relative to workspace or absolute)
            max_lines: Max lines to read
            
        Returns:
            ToolResult with file content
        """
        try:
            # Handle relative/absolute path
            if os.path.isabs(path):
                file_path = Path(path)
            else:
                file_path = self.workspace / path
                
            if not file_path.exists():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"File not found: {path}",
                )
                
            content = file_path.read_text()
            lines = content.split('\n')
            
            if len(lines) > max_lines:
                content = '\n'.join(lines[:max_lines])
                content += f"\n... (truncated, {len(lines)} total lines)"
                
            return ToolResult(success=True, output=content)
            
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=str(e),
            )
    
    def write_file(self, path: str, content: str) -> ToolResult:
        """
        Write content to a file
        
        Args:
            path: File path (relative to workspace)
            content: Content to write
            
        Returns:
            ToolResult
        """
        try:
            file_path = self.workspace / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            return ToolResult(
                success=True,
                output=f"Written to {path}",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=str(e),
            )
    
    def list_dir(self, path: str = ".") -> ToolResult:
        """
        List directory contents
        
        Args:
            path: Directory path
            
        Returns:
            ToolResult with file list
        """
        try:
            if os.path.isabs(path):
                dir_path = Path(path)
            else:
                dir_path = self.workspace / path
                
            if not dir_path.exists():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Directory not found: {path}",
                )
                
            items = []
            for item in sorted(dir_path.iterdir()):
                if item.is_dir():
                    items.append(f"[DIR]  {item.name}/")
                else:
                    size = item.stat().st_size
                    items.append(f"[FILE] {item.name} ({size} bytes)")
                    
            return ToolResult(
                success=True,
                output='\n'.join(items) if items else "(empty directory)",
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=str(e),
            )


# Tool definitions for LLM
TOOL_DEFINITIONS = [
    {
        "name": "run_code",
        "description": "Execute Python code. Use this to train models, process data, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute",
                }
            },
            "required": ["code"],
        },
    },
    {
        "name": "read_file",
        "description": "Read content of a file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path to read",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path to write",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write",
                }
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_dir",
        "description": "List contents of a directory",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path (default: current workspace)",
                }
            },
            "required": [],
        },
    },
    {
        "name": "submit",
        "description": "Submit trained model for evaluation. This will end the task.",
        "parameters": {
            "type": "object",
            "properties": {
                "model_path": {
                    "type": "string",
                    "description": "Path to the trained model",
                }
            },
            "required": ["model_path"],
        },
    },
]

