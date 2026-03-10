# hardshell/simulation/moltbook_server.py
import subprocess
import time
import requests
import os

class OfficialMoltbookSandbox:
    def __init__(self, api_path="external/moltbook_api"):
        self.api_path = os.path.abspath(api_path)
        self.process = None
        # The official repo uses /api/v1 routing
        self.api_url = "http://127.0.0.1:3000/api/v1" 

    def start(self):
        """Installs dependencies and launches the official Moltbook API."""
        print("Setting up official Moltbook Sandbox...")
        
        # Ensure dependencies are installed (runs npm install)
        if not os.path.exists(os.path.join(self.api_path, "node_modules")):
            print("Running npm install in external/moltbook_api...")
            subprocess.run(["npm", "install"], cwd=self.api_path, check=True)

        print("Launching API server...")
        # Usually 'npm run dev' or 'npm start' for their repo
        self.process = subprocess.Popen(
            ["npm", "run", "dev"], 
            cwd=self.api_path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Poll until the official API is responsive
        for _ in range(15):
            try:
                # Polling a standard endpoint to ensure it's up
                if requests.get(f"{self.api_url}/health").status_code == 200:
                    print("Official Moltbook API is live!")
                    return
            except requests.ConnectionError:
                time.sleep(2)
                
        raise RuntimeError("Official Moltbook API failed to start. Check Node/NPM installation.")

    def stop(self):
        if self.process:
            self.process.terminate()
            print("Moltbook Sandbox shut down.")