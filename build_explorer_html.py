"""
Rebuilds deeponet_explorer.html using whatever weights are currently in
deeponet_weights.json (produced by part5_deeponet_train_export.py).

Usage:
    python part5_deeponet_train_export.py   # trains + writes deeponet_weights.json
    python build_explorer_html.py           # rebuilds deeponet_explorer.html
"""

with open('deeponet_weights.json') as f:
    weights_json = f.read()

with open('deeponet_explorer_template.html') as f:
    template = f.read()

output = template.replace('__WEIGHTS_JSON__', weights_json)

with open('deeponet_explorer.html', 'w') as f:
    f.write(output)

print("deeponet_explorer.html rebuilt using deeponet_weights.json")
