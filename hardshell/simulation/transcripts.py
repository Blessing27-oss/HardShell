"""JSONL append-only loggers."""
# hardshell/simulation/transcripts.py
import json
import asyncio
from pathlib import Path
from typing import Dict, Any

class JSONLLogger:
    """
    Thread-safe, append-only logger for storing simulation transcripts.
    Guarantees no data corruption during highly concurrent async execution.
    """
    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        self.lock = asyncio.Lock()
        
        # Ensure the parent logs/ directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    async def append(self, record: Dict[str, Any]):
        """
        Safely appends a single dictionary as a JSON line.
        """
        # The lock forces concurrent workers to line up single-file just for the write operation
        async with self.lock:
            with open(self.file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + '\n')