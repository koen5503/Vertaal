import urllib.request
import json
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("RUNPOD_API_KEY")
URL = f"https://api.runpod.io/graphql?api_key={API_KEY}"

# De gegevens uit je vorige discovery
TEMPLATE_ID = "q1w7juynsy"
VOLUME_ID = "gl6ihbimas"
GPU_ID = "NVIDIA GeForce RTX 3090"
DC_ID = "EU-RO-1"

# Using the new podFindAndDeployOnDemand mutation
mutation = f'''
mutation {{
  podFindAndDeployOnDemand(input: {{
    gpuTypeId: "{GPU_ID}",
    templateId: "{TEMPLATE_ID}",
    networkVolumeId: "{VOLUME_ID}",
    dataCenterId: "{DC_ID}",
    gpuCount: 1
  }}) {{
    id
    desiredStatus
  }}
}}
'''

print("\n--- Testing RunPod podFindAndDeployOnDemand ---")
headers = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
data = json.dumps({"query": mutation}).encode('utf-8')
req = urllib.request.Request(URL, data=data, headers=headers, method="POST")

try:
    with urllib.request.urlopen(req, timeout=15) as f:
        res = json.loads(f.read().decode('utf-8'))
        print(json.dumps(res, indent=2))
except urllib.error.HTTPError as e:
    print(f"HTTP Error: {e.code}")
    print(e.read().decode())
except Exception as e:
    print(f"System Error: {e}")
