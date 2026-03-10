"""Minibook state reset & seeding."""
# hardshell/simulation/environment.py
import random
from typing import List, Dict

def generate_trial_timeline(benign_posts: List[str], payload: str) -> List[Dict[str, str]]:
    """
    Generates an in-memory Moltbook timeline for a single trial. 
    Completely bypasses SQLite to allow safe, lock-free async concurrency.
    """
    # Create the baseline benign posts
    timeline = [{"author": "team_member", "content": post} for post in benign_posts]
    
    # Inject the malicious payload at a random position to prevent positional bias
    insert_idx = random.randint(0, len(timeline))
    timeline.insert(insert_idx, {"author": "external_user", "content": payload})
    
    return timeline