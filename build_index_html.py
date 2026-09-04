"""
Updates index.html's embedded model weights using whatever is currently in
deeponet_weights.json (produced by part5_deeponet_train_export.py).

Usage:
    python part5_deeponet_train_export.py   # trains + writes deeponet_weights.json
    python build_explorer_html.py           # updates index.html in place
"""

with open('deeponet_weights.json') as f:
    weights_json = f.read()

with open('index.html') as f:
    html = f.read()

start = html.find('const WEIGHTS = ') + len('const WEIGHTS = ')
end = html.find(';\n', start)

updated_html = html[:start] + weights_json + html[end:]

with open('index.html', 'w') as f:
    f.write(updated_html)

print("index.html updated with weights from deeponet_weights.json")
