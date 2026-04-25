import urllib.request
import json
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("RUNPOD_API_KEY")
URL = f"https://api.runpod.io/graphql?api_key={API_KEY}"

def run_query(query_name, query_str):
    print(f"\n--- Testing: {query_name} ---")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    data = json.dumps({"query": query_str}).encode('utf-8')
    req = urllib.request.Request(URL, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as f:
            res = json.loads(f.read().decode('utf-8'))
            print(json.dumps(res, indent=2))
            return res
    except Exception as e:
        print(f"Error: {e}")
        return None

print("\n--- RunPod Brute-Force Debug Tool ---")

if not API_KEY:
    print("Fout: Geen RUNPOD_API_KEY gevonden in .env!")
    exit()

# Test 1: Simple Myself check
run_query("Basic Myself", "query { myself { id email } }")

# Test 2: Volumes (We know this one works)
run_query("Volumes", "query { myself { networkVolumes { id name dataCenterId } } }")

# Test 3: Templates (Trying a different structure)
run_query("Templates", "query { myself { templates { id name } } }")

# Test 4: All Pods
run_query("Active Pods", "query { myself { pods { id templateId } } }")

# Test 5: GPU IDs
run_query("GPU Types", "query { gpuTypes { id displayName } }")

print("\n--- Done ---")
