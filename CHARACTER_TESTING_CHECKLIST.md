# Character Consistency System - Testing Checklist

## Pre-Testing Setup

- [ ] All files saved and no syntax errors
- [ ] `characters.json` does not exist yet (will be auto-created)
- [ ] Both `main.py` and `web_app.py` are not running

## Test 1: Console App - First Character Creation

### Steps:
1. [ ] Run `python ollama_vision/main.py`
2. [ ] See "Character Selection" menu
3. [ ] Verify "No saved characters yet" or empty list
4. [ ] Press `[N]` to create new character
5. [ ] Enter prompt: `professional businessman, 40s, short gray hair, blue suit`
6. [ ] Wait for generation to complete
7. [ ] See "Save as Character?" prompt
8. [ ] Enter `y` to save
9. [ ] Enter character name: `Test Character 1`
10. [ ] Enter `y` to use generated image as reference
11. [ ] Verify success message with character ID
12. [ ] Check `characters.json` created in `ollama_vision/` folder
13. [ ] Open `characters.json` and verify character exists

### Expected Results:
- ✅ Character menu displays correctly
- ✅ Image generates successfully
- ✅ Character save workflow completes
- ✅ `characters.json` file created
- ✅ Character data properly formatted in JSON

## Test 2: Console App - Reuse Saved Character

### Steps:
1. [ ] Run `python ollama_vision/main.py` again
2. [ ] See character menu with "Test Character 1" listed
3. [ ] Select `[1]` to load character
4. [ ] Verify character info displays (description, seed)
5. [ ] Enter scene prompt: `sitting at desk in modern office`
6. [ ] Verify full combined prompt shows
7. [ ] Wait for generation
8. [ ] Check metadata .txt file includes character name

### Expected Results:
- ✅ Saved character appears in menu
- ✅ Character loads successfully
- ✅ Seed and settings applied correctly
- ✅ Combined prompt works
- ✅ Generated image matches original character
- ✅ Metadata file includes character info

## Test 3: Console App - Multiple Characters

### Steps:
1. [ ] Run `python ollama_vision/main.py`
2. [ ] Select `[N]` for new character
3. [ ] Enter different prompt: `young woman, long red hair, green eyes`
4. [ ] Generate and save as `Test Character 2`
5. [ ] Run app again
6. [ ] Verify both characters listed
7. [ ] Test selecting each character
8. [ ] Verify characters are distinct

### Expected Results:
- ✅ Multiple characters can be created
- ✅ Each has unique ID
- ✅ Both appear in selection menu
- ✅ Each maintains separate seed/settings
- ✅ Can switch between characters

## Test 4: Web App - Character Library Display

### Steps:
1. [ ] Start web app: `python ollama_vision/web_app.py`
2. [ ] Open browser to `http://localhost:3300`
3. [ ] Locate Character Library panel
4. [ ] Verify both test characters appear as cards
5. [ ] Check character names display correctly
6. [ ] Verify usage count shows (0 or 1 times)
7. [ ] Check if reference images display

### Expected Results:
- ✅ Character library panel visible
- ✅ Character cards render in grid
- ✅ Character names match saved data
- ✅ Usage stats display
- ✅ Reference images show (if paths correct)
- ✅ Empty state handles correctly if no characters

## Test 5: Web App - Character Selection

### Steps:
1. [ ] Click on "Test Character 1" card
2. [ ] Verify card gets green border (selected state)
3. [ ] Check "Selected: Test Character 1" indicator appears
4. [ ] Verify Character Consistency toggle auto-enables
5. [ ] Click different character card
6. [ ] Verify selection switches
7. [ ] Click "✕" to clear selection
8. [ ] Verify selection clears

### Expected Results:
- ✅ Click selects character
- ✅ Visual feedback (green border)
- ✅ Selected indicator appears
- ✅ Can switch selections
- ✅ Can clear selection
- ✅ Toggle auto-enables

## Test 6: Web App - Generate with Character

### Steps:
1. [ ] Select "Test Character 1"
2. [ ] Enter scene prompt: `video call on laptop`
3. [ ] Click "Generate Image"
4. [ ] Monitor job queue
5. [ ] Wait for completion
6. [ ] Compare result to reference image
7. [ ] Check metadata .txt file includes character info
8. [ ] Verify character usage count increments

### Expected Results:
- ✅ Job created successfully
- ✅ Image generates with character's seed
- ✅ Result matches character appearance
- ✅ Metadata includes character name and ID
- ✅ Usage counter increments in characters.json
- ✅ last_used timestamp updates

## Test 7: Web App - Character Refresh

### Steps:
1. [ ] Manually edit `characters.json` (add test property)
2. [ ] Click refresh button (🔄) in web UI
3. [ ] Verify character list updates
4. [ ] Or create character via console app
5. [ ] Click refresh in web UI
6. [ ] Verify new character appears

### Expected Results:
- ✅ Refresh button works
- ✅ Character list updates without page reload
- ✅ Changes from console app appear after refresh

## Test 8: Web App - Character Deletion

### Steps:
1. [ ] Click delete button (🗑️) on "Test Character 2"
2. [ ] Verify confirmation dialog appears
3. [ ] Click "Cancel" first time
4. [ ] Verify character NOT deleted
5. [ ] Click delete again
6. [ ] Click "OK" to confirm
7. [ ] Verify character removed from UI
8. [ ] Check `characters.json` - character removed
9. [ ] Refresh page - confirm still gone

### Expected Results:
- ✅ Delete button triggers confirmation
- ✅ Cancel preserves character
- ✅ Confirm deletes character
- ✅ UI updates immediately
- ✅ JSON file updated
- ✅ Deletion persists after refresh

## Test 9: API Endpoints

### Test GET /api/characters:
```bash
curl http://localhost:3300/api/characters
```
- [ ] Returns JSON with success: true
- [ ] Lists all characters
- [ ] Includes all character properties

### Test GET /api/characters/{id}:
```bash
# Get character ID from characters.json
curl http://localhost:3300/api/characters/REPLACE_WITH_REAL_ID
```
- [ ] Returns specific character
- [ ] Includes full character data
- [ ] Updates usage tracking

### Test POST /api/characters:
```bash
curl -X POST http://localhost:3300/api/characters \
  -H "Content-Type: application/json" \
  -d '{
    "name": "API Test Character",
    "description": "Created via API",
    "seed": 9999,
    "settings": {"num_steps": 50, "guidance_scale": 10}
  }'
```
- [ ] Returns success with character_id
- [ ] Character appears in UI after refresh
- [ ] Character saved in characters.json

### Test DELETE /api/characters/{id}:
```bash
curl -X DELETE http://localhost:3300/api/characters/REPLACE_WITH_REAL_ID
```
- [ ] Returns success message
- [ ] Character removed from JSON
- [ ] Character removed from UI after refresh

### Test POST /generate with character:
```bash
# Get character ID from characters.json
curl -X POST http://localhost:3300/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "walking in park",
    "character_id": "REPLACE_WITH_REAL_ID"
  }'
```
- [ ] Returns job_id
- [ ] Job uses character's seed
- [ ] Generated image matches character

## Test 10: Consistency Verification

### Steps:
1. [ ] Select same character in console app
2. [ ] Generate 3 different scenes:
   - `standing in office`
   - `sitting at cafe`
   - `walking in park`
3. [ ] Compare all 3 generated images
4. [ ] Verify character appearance is consistent
5. [ ] Check metadata files all show same seed
6. [ ] Verify different backgrounds/poses

### Expected Results:
- ✅ All images show same character
- ✅ Facial features match
- ✅ Overall appearance consistent
- ✅ Seed number identical in all metadata
- ✅ Scenes/poses are different

## Test 11: Error Handling

### Test Invalid Character ID:
```bash
curl http://localhost:3300/api/characters/invalid-id-12345
```
- [ ] Returns 404 error
- [ ] Error message: "Character not found"

### Test Missing Character Fields:
```bash
curl -X POST http://localhost:3300/api/characters \
  -H "Content-Type: application/json" \
  -d '{"name": "Incomplete"}'
```
- [ ] Returns 400 error
- [ ] Error message indicates missing fields

### Test Generate with Invalid Character:
1. [ ] Select character in web UI
2. [ ] Manually delete from characters.json
3. [ ] Try to generate
4. [ ] Verify graceful fallback (warning message)

## Test 12: File Integrity

### Verify characters.json:
1. [ ] Open `ollama_vision/characters.json`
2. [ ] Verify valid JSON syntax
3. [ ] Check all required fields present:
   - id, name, description, seed
   - settings, created_at
   - times_used, last_used
4. [ ] Verify reference_image paths are absolute
5. [ ] Check timestamps are ISO format

### Verify Metadata Files:
1. [ ] Navigate to latest generated image folder
2. [ ] Open .txt metadata file
3. [ ] Verify includes:
   - Character name (if used)
   - Character ID (if used)
   - Seed value
   - Generation settings

## Test 13: Usage Tracking

### Steps:
1. [ ] Note initial usage count for a character
2. [ ] Generate image using that character
3. [ ] Check characters.json
4. [ ] Verify times_used incremented by 1
5. [ ] Verify last_used timestamp updated
6. [ ] Generate again
7. [ ] Verify times_used incremented again
8. [ ] Check web UI shows updated count after refresh

### Expected Results:
- ✅ Usage counter increments correctly
- ✅ Timestamp updates to current time
- ✅ Counter persists across sessions
- ✅ Web UI reflects accurate counts

## Test 14: Cross-Platform (Console ↔ Web)

### Steps:
1. [ ] Create character in console app
2. [ ] Start web app
3. [ ] Verify character appears in web UI
4. [ ] Generate image using character in web UI
5. [ ] Check characters.json updated
6. [ ] Create character via web API
7. [ ] Run console app
8. [ ] Verify API-created character appears in menu

### Expected Results:
- ✅ Characters created in console appear in web
- ✅ Characters created in web appear in console
- ✅ Usage tracking works across both
- ✅ Single source of truth (characters.json)

## Test 15: Reference Images

### Steps:
1. [ ] Create character with reference image
2. [ ] Verify image path in characters.json is absolute
3. [ ] Check image file exists at that path
4. [ ] Open web UI
5. [ ] Verify reference image displays in character card
6. [ ] Move/delete reference image file
7. [ ] Refresh web UI
8. [ ] Verify graceful fallback (blank thumbnail)
9. [ ] Character still functional without reference

### Expected Results:
- ✅ Reference images save with absolute paths
- ✅ Images display in web UI
- ✅ Missing images don't break system
- ✅ Characters work without reference images

## Test 16: Edge Cases

### Empty Prompt with Character:
1. [ ] Select character
2. [ ] Leave scene prompt blank
3. [ ] Try to generate
- [ ] System should use character description only

### Special Characters in Names:
1. [ ] Create character with name: `Test "Character" #3`
2. [ ] Verify saves correctly
3. [ ] Verify displays correctly in UI
- [ ] JSON escaping handled properly

### Very Long Description:
1. [ ] Create character with 500+ word description
2. [ ] Verify saves correctly
3. [ ] Verify doesn't break UI rendering
- [ ] Long text handled gracefully

### Concurrent Generations:
1. [ ] Start generation with character in web UI
2. [ ] Immediately start another with same character
3. [ ] Verify both use same seed
4. [ ] Verify both complete successfully
- [ ] No race conditions in usage tracking

## Test 17: Performance

### Load Test:
1. [ ] Create 20+ characters
2. [ ] Check web UI loading time
3. [ ] Check character selection responsiveness
4. [ ] Verify characters.json file size reasonable

### Expected Results:
- ✅ UI loads quickly even with many characters
- ✅ Character selection responsive
- ✅ JSON file remains manageable
- ✅ No performance degradation

## Post-Testing Cleanup

- [ ] Delete test characters if desired
- [ ] Keep 1-2 characters for future testing
- [ ] Verify backup of characters.json (optional)
- [ ] Document any issues found

## Issues Template

If you find issues, document them:

```
Issue #: ___
Test: ___
Steps to Reproduce:
1. 
2.
3.

Expected Behavior:


Actual Behavior:


Error Messages:


```

## Success Criteria

All tests should pass for system to be considered production-ready:
- [ ] All console app tests pass (Tests 1-3)
- [ ] All web app tests pass (Tests 4-8)
- [ ] All API tests pass (Test 9)
- [ ] Consistency verified (Test 10)
- [ ] Error handling works (Test 11)
- [ ] File integrity verified (Test 12)
- [ ] Usage tracking works (Test 13)
- [ ] Cross-platform works (Test 14)
- [ ] Reference images work (Test 15)
- [ ] Edge cases handled (Test 16)
- [ ] Performance acceptable (Test 17)

## Final Checklist

- [ ] All 17 test sections completed
- [ ] No critical issues found
- [ ] Documentation reviewed and accurate
- [ ] Example workflows verified
- [ ] Ready for production use

---

**Testing completed by:** _____________
**Date:** _____________
**Status:** ⬜ Pass | ⬜ Fail | ⬜ Pass with issues
**Notes:**
