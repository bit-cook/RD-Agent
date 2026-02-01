"""
Grading Server Client

统一的 grading server 提交函数，供 runner 和 agent 使用。
"""
import os
from typing import Optional

import requests
from loguru import logger


def submit_to_grading_server(
    model_path: str,
    grading_url: Optional[str] = None,
    timeout: int = 600,
) -> dict | None:
    """
    提交模型到 grading server 评测
    
    Args:
        model_path: 模型路径
        grading_url: grading server URL，默认从环境变量 GRADING_SERVER_URL 获取
        timeout: 请求超时时间（秒）
        
    Returns:
        评测结果 dict，失败返回 None
    """
    url = grading_url or os.environ.get("GRADING_SERVER_URL")
    if not url:
        return None
    
    try:
        logger.info(f"Submitting to grading server: {url}/submit")
        resp = requests.post(
            f"{url}/submit",
            json={"model_path": model_path},
            timeout=timeout,
        )
        if resp.status_code == 200:
            result = resp.json()
            score = result.get("score")
            improvement = result.get("improvement")
            logger.info(f"Grading result: score={score}, improvement={improvement}")
            return result
        else:
            logger.warning(f"Grading server returned {resp.status_code}: {resp.text}")
            return None
    except requests.Timeout:
        logger.warning(f"Grading server timeout after {timeout}s")
        return None
    except Exception as e:
        logger.warning(f"Grading server error: {e}")
        return None


def set_baseline_score(score: float, grading_url: Optional[str] = None) -> bool:
    """
    设置 baseline score
    
    Args:
        score: baseline 分数
        grading_url: grading server URL
        
    Returns:
        是否设置成功
    """
    url = grading_url or os.environ.get("GRADING_SERVER_URL")
    if not url:
        return False
    
    try:
        resp = requests.post(f"{url}/set_baseline", json={"score": score}, timeout=30)
        return resp.status_code == 200
    except Exception as e:
        logger.warning(f"Failed to set baseline score: {e}")
        return False


def get_best_score(grading_url: Optional[str] = None) -> dict | None:
    """
    获取最高分
    
    Returns:
        {"best": {...}, "total_submissions": N} 或 None
    """
    url = grading_url or os.environ.get("GRADING_SERVER_URL")
    if not url:
        return None
    
    try:
        resp = requests.get(f"{url}/best", timeout=30)
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception as e:
        logger.warning(f"Failed to get best score: {e}")
        return None
