# Character Consistency System Guide

## Overview

The Character Consistency System allows you to save characters with their exact generation parameters and reuse them across different scenes, poses, and situations while maintaining perfect visual consistency.

## How It Works

### Core Concept
- **Fixed Seed + Character Description = Consistent Character**
- When you save a character, the system stores:
  - Character name
  - Complete character description (optimized prompt)
  - Fixed seed value
  - Generation settings (steps, guidance scale)
  - Optional reference image

### Character Generation Workflow

```
1. Generate Initial Character
   ↓
2. Save as Character (with name and reference image)
   ↓
3. Reuse Character with Different Scenes
   ↓
4. Generate Consistent Results Every Time
```

## Using the Console App (main.py)

### Step 1: Character Selection Menu

When you run `python ollama_vision/main.py`, you'll see:

```
═══ Character Selection ═══

Saved Characters:
1. John Smith - Used 3 times (Last: 2024-01-15)
2. Sarah Connor - Used 5 times (Last: 2024-01-14)

Options:
  [1-N] - Select saved character by number
  [N]   - Create new character
  [Enter] - Skip (no character consistency)
Choose option:
```

**Options:**
- **[1-N]**: Select existing character by number
- **[N]**: Create new character (will prompt to save after generation)
- **[Enter]**: Skip character selection (random generation)

### Step 2: Using Existing Character

If you select a saved character:

```
✓ Loaded character: John Smith
Description: A middle-aged man with short brown hair, blue eyes...
Fixed seed: 42

Character loaded: John Smith
Tip: Describe the scene/pose for this character
Enter scene/pose description: standing in a forest, sunset lighting
```

The system will:
- Combine character description with your scene
- Use the character's saved seed
- Apply character's generation settings
- Generate consistent character in the new scene

### Step 3: Creating New Character

If you choose to create new character [N]:

```
Enter your prompt: beautiful woman with long red hair, green eyes

[After generation completes]

═══ Save as Character? ═══
Save this as a character for future use? (y/N): y
Character name: Emma
Use first generated image as reference? (Y/n): y

✓ Character 'Emma' saved! (ID: abc123)
Seed: 42, Settings: 60 steps, guidance 12
```

## Using the Web App (web_app.py)

### Character Library Panel

The web interface includes a visual Character Library:

```
🎭 Character Library                    🔄

┌─────────┐  ┌─────────┐  ┌─────────┐
│         │  │         │  │         │
│ [Image] │  │ [Image] │  │ [Image] │
│         │  │         │  │         │
│ John    │  │ Sarah   │  │ Emma    │
│ Used 3x │  │ Used 5x │  │ Used 2x │
└─────────┘  └─────────┘  └─────────┘

Selected: John Smith                   ✕
```

### Workflow

1. **Browse Characters**: Scroll through saved characters with preview images
2. **Select Character**: Click on a character card to select it
3. **Enter Scene**: Type scene/pose description in prompt field
4. **Generate**: Click "Generate Image" - system uses character's seed and settings
5. **Delete Characters**: Click 🗑️ button on character card

### Selected Character Indicator

When a character is selected:
- Green border highlights the selected card
- "Selected: [Character Name]" appears below the library
- Character Consistency toggle automatically enables
- Character's saved settings are applied

## API Endpoints (for developers)

### GET /api/characters
List all saved characters

**Response:**
```json
{
  "success": true,
  "characters": [
    {
      "id": "abc123",
      "name": "John Smith",
      "description": "A middle-aged man with...",
      "seed": 42,
      "settings": {
        "num_steps": 60,
        "guidance_scale": 12,
        "model": "realvisxl"
      },
      "reference_image": "/path/to/image.png",
      "created_at": "2024-01-15T10:30:00",
      "times_used": 3,
      "last_used": "2024-01-15T14:20:00"
    }
  ]
}
```

### GET /api/characters/{character_id}
Get specific character by ID

**Response:**
```json
{
  "success": true,
  "character": { /* character object */ }
}
```

### POST /api/characters
Create new character

**Request:**
```json
{
  "name": "Emma Watson",
  "description": "Beautiful woman with long red hair...",
  "seed": 42,
  "settings": {
    "num_steps": 60,
    "guidance_scale": 12,
    "model": "realvisxl"
  },
  "reference_image": "/path/to/image.png"
}
```

**Response:**
```json
{
  "success": true,
  "character_id": "xyz789",
  "message": "Character 'Emma Watson' created successfully"
}
```

### DELETE /api/characters/{character_id}
Delete a character

**Response:**
```json
{
  "success": true,
  "message": "Character deleted successfully"
}
```

### POST /generate (with character)
Generate image using a character

**Request:**
```json
{
  "prompt": "standing in a forest",
  "character_id": "abc123",
  "optimize_prompt": true,
  "character_consistency": true,
  "guidance_scale": 12,
  "inference_steps": 60
}
```

## Character Storage

Characters are stored in `ollama_vision/characters.json`:

```json
{
  "abc123": {
    "id": "abc123",
    "name": "John Smith",
    "description": "A middle-aged man with short brown hair...",
    "seed": 42,
    "settings": {
      "num_steps": 60,
      "guidance_scale": 12,
      "model": "realvisxl"
    },
    "reference_image": "/Users/.../generated_20240115_143000.png",
    "created_at": "2024-01-15T10:30:00",
    "times_used": 3,
    "last_used": "2024-01-15T14:20:00"
  }
}
```

## Best Practices

### Creating Great Characters

1. **Detailed Initial Prompt**: Provide comprehensive character description
   ```
   Good: "A 30-year-old woman with long wavy blonde hair, bright blue eyes, 
          fair skin, wearing a red dress, professional photoshoot style"
   
   Poor: "woman"
   ```

2. **Use Prompt Optimizer**: Let Ollama enhance your description for consistency

3. **Save Reference Image**: Always save the generated image as reference

4. **Test Consistency**: Generate 2-3 variations immediately to verify consistency

### Reusing Characters

1. **Scene-Only Prompts**: When using saved character, only describe the scene
   ```
   Character Selected: John Smith
   Prompt: "standing in a modern office, smiling at camera"
   ```

2. **Keep Settings Consistent**: Character's saved settings are automatically applied

3. **Track Usage**: Monitor how many times you've used a character

### Managing Characters

- **Descriptive Names**: Use clear, memorable names ("Professional John" vs "Character 1")
- **Regular Cleanup**: Delete unused or low-quality characters
- **Backup characters.json**: Important characters should be backed up
- **Test Before Deleting**: Generate one final image before removing a character

## Troubleshooting

### Character Looks Different

**Problem**: Reused character doesn't match reference
**Solutions**:
1. Verify seed is identical (check .txt metadata file)
2. Ensure same model is being used (RealVisXL V5.0)
3. Check that settings match (steps, guidance scale)
4. Verify character description wasn't modified

### Can't Load Character

**Problem**: Character selection doesn't work
**Solutions**:
1. Check `characters.json` exists in `ollama_vision/` folder
2. Verify JSON syntax is valid
3. Ensure character ID exists in the file
4. Restart web app or console app

### Reference Image Missing

**Problem**: Character card shows blank thumbnail
**Solutions**:
1. Image file may have been moved/deleted
2. Path in `characters.json` may be incorrect
3. Character still works without reference image
4. Update path or regenerate reference image

## Advanced Usage

### Programmatic Character Creation

```python
from character_manager import CharacterManager

manager = CharacterManager()

# Save character
char_id = manager.save_character(
    name="Custom Character",
    description="Detailed character description...",
    seed=12345,
    settings={
        'num_steps': 60,
        'guidance_scale': 12,
        'model': 'realvisxl'
    },
    reference_image="/path/to/image.png"
)

# Retrieve and use
character = manager.get_character(char_id)
prompt = manager.generate_character_prompt(char_id, "in a forest")
```

### Batch Character Operations

```python
# List all characters
characters = manager.list_characters()

# Find characters by usage
frequent_chars = [c for c in characters if c['times_used'] > 10]

# Delete unused characters
for char in characters:
    if char['times_used'] == 0:
        manager.delete_character(char['id'])
```

## Examples

### Example 1: Creating a Consistent Character

```
Step 1: Initial Generation
Prompt: "professional businessman, 40s, short gray hair, glasses"
Settings: 60 steps, guidance 12
Result: Perfect character generated

Step 2: Save Character
Name: "Corporate Dave"
Seed: 42
Reference: First generated image

Step 3: Reuse with Different Scenes
Scene 1: "in a boardroom presenting"
Scene 2: "drinking coffee in office"
Scene 3: "video call on laptop"
All results: Same character, different poses
```

### Example 2: Character Evolution Workflow

```
Week 1: Create "Sarah the Warrior"
- Generate base character
- Save with seed 100
- Use in 5 different battle scenes

Week 2: Refine Character
- Keep same seed (100)
- Enhance character description
- Update reference image
- Generate more consistent results

Week 3: Character Library
- Have 10+ saved characters
- Mix and match in group scenes
- All maintain consistency
```

## Summary

The Character Consistency System provides:
- ✅ Perfect character consistency across generations
- ✅ Efficient character reuse with different scenes
- ✅ Visual character library management
- ✅ Both console and web interface support
- ✅ Comprehensive tracking and metadata

**Key Insight**: Fixed seed + identical prompt = identical character. This system automates that workflow for effortless character consistency.
