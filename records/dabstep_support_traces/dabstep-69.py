import pandas as pd
import json

# Read fees.json
with open('fees.json', 'r') as f:
    fees_data = json.load(f)

# Extract monthly_volumes
volumes = []
for fee in fees_data:
    vol = fee.get('monthly_volume')
    if vol is not None:
        volumes.append(vol)

# Find the highest volume
if volumes:
    highest_volume = max(volumes)
    # The answer should be the string representing the volume
    answer = str(highest_volume)
else:
    answer = 'Not Applicable'

print(json.dumps({'answer': answer}))