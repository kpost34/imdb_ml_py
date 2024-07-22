# minify_json.py

import json

# Read the original JSON file
with open('docs/app.json', 'r') as f:
    data = json.load(f)

# Write minified JSON to a new file
with open('docs/app_minified.json', 'w') as f:
    json.dump(data, f, separators=(',', ':'))

#do the following after each update to the app (that's committed/pushed):
# Export Shiny app
# shinylive export . docs

# Minify JSON file
# python minify_json.py

# Serve the docs folder
# python -m http.server --directory docs --bind localhost 8008

# Commit/push the docs folder to GitHub


