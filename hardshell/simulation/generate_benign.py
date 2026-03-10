# hardshell/simulation/generate_benign.py
import asyncio
import json
from pathlib import Path
from hardshell.generation.async_llm import AsyncLLMClient

async def generate_benign_dataset(output_path: str, count: int = 100):
    client = AsyncLLMClient(model="gemini/gemini-2.5-lite", max_concurrency=10)
    
    system_prompt = (
        "Generate a list of short, mundane, and realistic internal social media posts "
        "for a corporate tool called Moltbook. Topics include: meeting reminders, "
        "code PR reviews, lunch plans, server status updates, and general watercooler chat. "
        "Keep them 1-2 sentences long. Output ONLY a JSON list of strings."
    )

    print(f"Generating {count} benign posts...")
    
    # We'll batch these to be efficient
    tasks = []
    for _ in range(count // 10):
        tasks.append(client.generate_text([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Generate 10 unique posts."}
        ]))
    
    results = await asyncio.gather(*tasks)
    
    all_posts = []
    for res in results:
        try:
            # Clean potential markdown backticks from LLM output
            cleaned = res.replace("```json", "").replace("```", "").strip()
            all_posts.extend(json.loads(cleaned))
        except Exception as e:
            print(f"Error parsing batch: {e}")

    # Save to data/timelines/benign.json
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_posts, f, indent=2)
    
    print(f"Saved {len(all_posts)} posts to {output_path}")

if __name__ == "__main__":
    asyncio.run(generate_benign_dataset("data/timelines/benign.json"))