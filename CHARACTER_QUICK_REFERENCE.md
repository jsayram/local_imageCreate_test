# Character Consistency - Quick Reference

## 🎯 What It Does
Save characters with exact settings and reuse them in different scenes with perfect consistency.

## 🚀 Quick Start

### Console App (main.py)
```bash
python ollama_vision/main.py
```

**First Time - Create Character:**
1. Select `[N]` for new character
2. Enter prompt: `professional businessman, 40s, gray hair`
3. After generation: `y` to save
4. Name: `Corporate Dave`
5. Use reference image: `y`

**Reuse Character:**
1. Select `[1]` (or character number)
2. Enter scene only: `sitting at desk in modern office`
3. Generate → Same character, new scene!

### Web App (web_app.py)
```bash
python ollama_vision/web_app.py
# Open http://localhost:3300
```

**Use Character:**
1. Click character card in library
2. Enter scene in prompt
3. Click "Generate Image"

## 🔑 Key Concepts

**Fixed Seed = Consistent Character**
- Same seed + same prompt = identical character
- System stores seed with character
- Automatically applies when you select character

**Character = Description + Seed + Settings**
```json
{
  "name": "John",
  "description": "Full character description",
  "seed": 42,
  "settings": {"steps": 60, "guidance": 12}
}
```

## 📋 Console Commands

### Character Selection Menu
```
[1-N]   → Select character by number
[N]     → Create new character  
[Enter] → Skip (no character)
```

### After Generation
```
y → Save as character
n → Skip saving
```

## 🌐 Web Interface

### Character Library Panel
- **Grid of cards** with character thumbnails
- **Click card** to select
- **🔄 button** to refresh list
- **🗑️ button** to delete character
- **Green border** = selected character

### When Character Selected
- Auto-enables Character Consistency toggle
- Prompt becomes scene description only
- Uses character's saved seed and settings

## 🛠️ API Endpoints

```bash
# List all characters
GET /api/characters

# Get specific character
GET /api/characters/{id}

# Create character
POST /api/characters
{
  "name": "Character Name",
  "description": "Full description",
  "seed": 42,
  "settings": {"num_steps": 60, "guidance_scale": 12}
}

# Delete character
DELETE /api/characters/{id}

# Generate with character
POST /generate
{
  "prompt": "scene description",
  "character_id": "abc123"
}
```

## ✅ Best Practices

**Creating Characters:**
- ✅ Detailed initial description
- ✅ Enable prompt optimizer
- ✅ Save reference image
- ✅ Test with 2-3 scenes immediately

**Reusing Characters:**
- ✅ Scene-only prompts
- ✅ Keep settings consistent
- ✅ Use descriptive character names
- ✅ Track usage for cleanup

**Example Prompts:**

**Initial Character:**
```
professional woman, 30s, long brown hair, blue suit, 
confident expression, corporate headshot style
```

**Scene Reuse:**
```
standing in boardroom presenting
sitting at desk typing on laptop
video call on computer screen
walking through modern office
```

## 📁 File Locations

```
ollama_vision/
├── character_manager.py     # Core system
├── characters.json          # Character storage (auto-created)
├── main.py                  # Console app (modified)
└── web_app.py              # Web app (modified)

/
├── CHARACTER_SYSTEM_GUIDE.md      # Full documentation
└── IMPLEMENTATION_SUMMARY.md      # Technical details
```

## 🐛 Troubleshooting

**Character looks different:**
- Check seed matches in metadata .txt file
- Verify using same model (RealVisXL V5.0)
- Ensure settings match

**Character won't load:**
- Check `characters.json` exists
- Verify valid JSON syntax
- Restart app

**No characters showing:**
- Click refresh button (🔄)
- Check `characters.json` not empty
- Verify no JSON errors

## 💡 Pro Tips

1. **Name Convention**: Use descriptive names
   - ✅ "Professional John - Business"
   - ❌ "Character 1"

2. **Scene Prompts**: Keep them focused
   - ✅ "standing in forest, sunset lighting"
   - ❌ "man standing in forest with birds flying"

3. **Reference Images**: Always save them
   - Helps identify character visually
   - Confirms consistency later

4. **Regular Cleanup**: Delete unused characters
   - Check usage count
   - Remove failed experiments

## 🎨 Workflow Examples

### Example 1: Portrait Series
```
1. Create character: "professional headshot, businessman"
2. Save as "CEO John"
3. Generate scenes:
   - "neutral background, direct eye contact"
   - "office background, smiling"
   - "outdoor setting, casual expression"
All use same seed → consistent person
```

### Example 2: Story Illustrations
```
1. Create character: "young adventurer, explorer outfit"
2. Save as "Explorer Emma"
3. Generate story scenes:
   - "examining ancient map in library"
   - "climbing mountain cliff"
   - "discovering hidden temple"
Same character across narrative
```

## 📊 Character Metadata

Each character stores:
- **name**: Display name
- **description**: Full optimized prompt
- **seed**: Fixed seed number
- **settings**: Generation parameters
- **reference_image**: Path to image
- **created_at**: Creation timestamp
- **times_used**: Usage counter
- **last_used**: Last usage timestamp

## 🔗 Related Features

Works with:
- ✅ Character Consistency toggle
- ✅ Advanced Settings (steps, guidance)
- ✅ Prompt Optimizer
- ✅ Job Queue system
- ✅ Per-image folders with metadata

## 📖 Documentation

- **USER GUIDE**: `CHARACTER_SYSTEM_GUIDE.md`
- **TECH DOCS**: `IMPLEMENTATION_SUMMARY.md`
- **THIS FILE**: Quick reference

---

**Need help?** Check `CHARACTER_SYSTEM_GUIDE.md` for detailed documentation.
