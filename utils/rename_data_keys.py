import json

# after preparing the dataset, rename keys for convinience
with open('sdata.json', 'r') as file:
    data = json.load(file)
key_mapping = {'p': 'instruction', 'c': 'input', 'l': 'output'}
for d in data:
    for key, new_key in key_mapping.items():
        if key in d:
            d[new_key] = d.pop(key)
with open('isdata.json', 'w') as file:
    json.dump(data, file, indent=4)
