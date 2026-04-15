# Asset Acquisition & Extraction Pipeline

## Sources

| Source | Assets | License | API/Bulk |
|---|---|---|---|
| Poly Haven | HDRIs, textures, models | CC0 | Yes (REST API) |
| AmbientCG | Textures | CC0 | Yes (REST API) |
| BlenderKit | Models, materials | CC0/free | Blender addon only |
| Blender Studio | Full scenes | CC | Manual download |
| Sketchfab | Models | All free | Yes (REST API) |

---

## 1. Poly Haven — Automated Bulk Download

REST API, no auth required.

```bash
# Fetch all HDRI asset IDs
curl "https://api.polyhaven.com/assets?type=hdris" | jq 'keys[]' > hdri_list.txt

# Download all HDRIs at 4K resolution
while read id; do
  url=$(curl -s "https://api.polyhaven.com/files/$id" | jq -r '.hdri."4k".hdr.url')
  wget -P data/hdri/ "$url"
done < hdri_list.txt
```

Same pattern for textures (`type=textures`) and models (`type=models`).

Texture file keys to pull per asset: `diffuse`, `rough`, `nor_gl`, `disp` — all at `2k` or `4k`.

---

## 2. AmbientCG — Automated Bulk Download

```bash
# Fetch asset list (paginated, 100 per page)
curl "https://ambientcg.com/api/v2/full_json?type=Material&sort=Latest&limit=100&offset=0" \
  | jq '.foundAssets[].assetId' > ambientcg_ids.txt

# Download ZIP for each asset (contains all PBR maps)
while read id; do
  wget "https://ambientcg.com/get?file=${id}_2K-JPG.zip" -P data/textures/ambientcg/
  unzip -q "data/textures/ambientcg/${id}_2K-JPG.zip" -d "data/textures/ambientcg/$id/"
done < ambientcg_ids.txt
```

---

## 3. BlenderKit — In-Blender Only

No public API for bulk download. Access via the Blender addon at render time.

- Install: `Edit > Preferences > Add-ons > BlenderKit`
- Free account required
- Assets are cached locally after first use: `~/.local/share/blenderkit_data/`
- Script can query the local cache once assets are downloaded interactively

---

## 4. Blender Studio Open Projects — Manual

No API. Download `.blend` files directly from each project page:

- Sprite Fright, Cosmos Laundromat, Charge, Elixir, etc.
- Extract and catalog manually into `data/scenes/blender_studio/`
- These serve as full scene templates, not individual assets

---

## 5. Sketchfab — Bulk Download (All Free/Downloadable)

Requires a free API token. Paginate through all downloadable models regardless of license.

```bash
TOKEN="your_token_here"
PAGE=1

# Paginate through all downloadable models (24 per page)
while true; do
  response=$(curl -s "https://api.sketchfab.com/v3/models?downloadable=true&count=24&page=$PAGE" \
    -H "Authorization: Token $TOKEN")
  
  echo "$response" | jq -r '.results[].uid' >> sketchfab_ids.txt
  
  # Stop if no next page
  next=$(echo "$response" | jq -r '.next')
  [ "$next" = "null" ] && break
  ((PAGE++))
done

# Download each model as GLTF
while read uid; do
  link=$(curl -s "https://api.sketchfab.com/v3/models/$uid/download" \
    -H "Authorization: Token $TOKEN" | jq -r '.gltf.url')
  [ "$link" != "null" ] && wget -q "$link" -P data/models/sketchfab/
done < sketchfab_ids.txt
```

---

## Local Directory Structure

```
data/
  hdri/                  # .hdr / .exr files from Poly Haven
  textures/
    polyhaven/           # per-asset folders with PBR maps
    ambientcg/           # per-asset folders with PBR maps
  models/
    polyhaven/           # .blend / .glb
    sketchfab/           # .glb zips, extracted
  scenes/
    blender_studio/      # full .blend scene files
```

---

## Key Decisions

- **Resolution**: 2K textures for variety/speed; 4K HDRIs for lighting quality
- **Format**: HDRIs as `.hdr`, textures as `.jpg` (diffuse) + `.png` (normal/rough/disp), models as `.glb`
- **Deduplication**: hash files post-download to drop duplicates across sources
- **Validation**: confirm each texture folder has at minimum: diffuse, roughness, normal map
