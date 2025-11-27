"""
Structured Logging System
Provides multi-language compatible, machine-readable logs for test execution.
"""
import json
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

from app import utils   # ONLY required import (for utils.logger)


class LogLevel(str, Enum):
    """Log level enumeration"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"
    ACTION = "ACTION"


class LogCategory(str, Enum):
    """Log category for organization"""
    INITIALIZATION = "INIT"
    PLAN_BUILDING = "PLAN"
    SEARCH = "SEARCH"
    GENERATION = "GENERATION"
    EXECUTION = "EXECUTION"
    HEALING = "HEALING"
    STORAGE = "STORAGE"
    CLEANUP = "CLEANUP"


@dataclass
class StructuredLog:
    """Machine-readable log entry"""
    timestamp: str
    level: str
    category: str
    message: str
    code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    duration_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    def to_readable(self) -> str:
        parts = [f"[{self.timestamp}]", f"[{self.level}]", f"[{self.category}]", self.message]
        if self.code:
            parts.append(f"(Code: {self.code})")
        if self.duration_ms:
            parts.append(f"({self.duration_ms:.2f}ms)")
        return " ".join(parts)


class StructuredLogger:
    """
    Structured logging for test execution.
    Maintains both human-readable and machine-readable logs.
    """
    
    def __init__(self, testcase_id: str):
        self.testcase_id = testcase_id
        self.logs: List[StructuredLog] = []
        self.start_time = datetime.now()
    
    def _get_timestamp(self) -> str:
        return datetime.now().isoformat()
    
    def log(self, level: LogLevel, category: LogCategory, message: str,
            code: Optional[str] = None, details: Optional[Dict[str, Any]] = None,
            duration_ms: Optional[float] = None):
        
        entry = StructuredLog(
            timestamp=self._get_timestamp(),
            level=level.value,
            category=category.value,
            message=message,
            code=code,
            details=details,
            duration_ms=duration_ms
        )
        
        self.logs.append(entry)

        # write to standard application logger
        level_map = {
            LogLevel.DEBUG: utils.logger.debug,
            LogLevel.INFO: utils.logger.info,
            LogLevel.WARNING: utils.logger.warning,
            LogLevel.ERROR: utils.logger.error,
            LogLevel.SUCCESS: utils.logger.info,
            LogLevel.ACTION: utils.logger.info,
        }
        log_func = level_map.get(level, utils.logger.info)
        log_func(entry.to_readable())

        return entry
    
    def info(self, category: LogCategory, message: str, code=None, details=None):
        return self.log(LogLevel.INFO, category, message, code, details)
    
    def error(self, category: LogCategory, message: str, code=None, details=None):
        return self.log(LogLevel.ERROR, category, message, code, details)
    
    def success(self, category: LogCategory, message: str, code=None, details=None):
        return self.log(LogLevel.SUCCESS, category, message, code, details)
    
    def action(self, action_name: str, status: str, duration_ms=None, details=None):
        msg = f"Action: {action_name} - Status: {status}"
        return self.log(LogLevel.ACTION, LogCategory.EXECUTION, msg,
                        code=action_name, details=details, duration_ms=duration_ms)
    
    def step_execution(self, step_name: str, step_number: int,
                       status: str, duration_ms: float, error=None,
                       screenshot=None, dom_snapshot=None):

        details = {
            "step_number": step_number,
            "status": status,
            "duration_ms": duration_ms
        }
        if error:
            details["error"] = error
        if screenshot:
            details["screenshot"] = screenshot
        if dom_snapshot:
            details["dom_snapshot_size"] = len(dom_snapshot)

        level = LogLevel.SUCCESS if status == "PASS" else LogLevel.ERROR
        
        return self.log(
            level,
            LogCategory.EXECUTION,
            f"Step {step_number}: {step_name}",
            code=f"STEP_{step_number:03d}",
            details=details,
            duration_ms=duration_ms
        )
    
    def get_readable_logs(self) -> str:
        return "\n".join([log.to_readable() for log in self.logs])
    
    def get_json_logs(self) -> str:
        return json.dumps([log.to_dict() for log in self.logs], indent=2)
    
    def get_summary(self) -> Dict[str, Any]:
        total_ms = (datetime.now() - self.start_time).total_seconds() * 1000

        error_count = sum(1 for log in self.logs if log.level == LogLevel.ERROR.value)
        success_count = sum(1 for log in self.logs if log.level == LogLevel.SUCCESS.value)
        
        return {
            "testcase_id": self.testcase_id,
            "total_logs": len(self.logs),
            "total_time_ms": total_ms,
            "success_count": success_count,
            "error_count": error_count,
            "status": "SUCCESS" if error_count == 0 else "FAILED"
        }


def extract_madl_from_logs(logs: List[StructuredLog]) -> Dict[str, Any]:
    """
    Extract MADL-relevant data from execution logs.
    Identifies successful actions that can become reusable methods.
    """
    madl_data = {
        "actions": [],
        "selectors": [],
        "keywords": set(),
        "flow_description": ""
    }

    successful = [log for log in logs if log.level == LogLevel.SUCCESS.value]

    for step in successful:
        if step.code:
            madl_data["actions"].append(step.code)

        words = step.message.lower().split()
        for w in words:
            if len(w) > 3:
                madl_data["keywords"].add(w)

    if madl_data["actions"]:
        madl_data["flow_description"] = " -> ".join(madl_data["actions"])

    madl_data["keywords"] = list(madl_data["keywords"])

    return madl_data
