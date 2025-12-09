"""
Character Management System for Consistent Image Generation
Allows saving and reusing character profiles with fixed seeds and descriptions.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional


class CharacterManager:
    """Manages saved character profiles for consistent generation."""
    
    def __init__(self, storage_file: str = "ollama_vision/characters.json"):
        self.storage_file = storage_file
        self.characters = self._load_characters()
    
    def _load_characters(self) -> Dict:
        """Load saved characters from JSON file."""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return {}
        return {}
    
    def _save_characters(self):
        """Save characters to JSON file."""
        os.makedirs(os.path.dirname(self.storage_file), exist_ok=True)
        with open(self.storage_file, 'w') as f:
            json.dump(self.characters, f, indent=2)
    
    def save_character(self, name: str, description: str, seed: int, 
                      settings: Dict = None, reference_image: str = None) -> str:
        """
        Save a character profile.
        
        Args:
            name: Unique character name/ID
            description: Physical description (face, hair, body, etc.)
            seed: The exact seed used to generate this character
            settings: Dict with steps, guidance, width, height, etc.
            reference_image: Path to reference image
            
        Returns:
            character_id: The unique ID for this character
        """
        import uuid
        character_id = str(uuid.uuid4())[:8]
        
        self.characters[character_id] = {
            'id': character_id,
            'name': name,
            'description': description,
            'seed': seed,
            'settings': settings or {},
            'reference_image': reference_image,
            'created_at': datetime.now().isoformat(),
            'last_used': datetime.now().isoformat(),
            'times_used': 0
        }
        
        self._save_characters()
        return character_id
    
    def get_character(self, character_id: str) -> Optional[Dict]:
        """Get a character profile by ID and update usage stats."""
        character = self.characters.get(character_id)
        if character:
            # Update last used timestamp and increment usage
            character['last_used'] = datetime.now().isoformat()
            character['usage_count'] = character.get('usage_count', 0) + 1
            self._save_characters()
        return character
    
    def list_characters(self) -> List[Dict]:
        """List all saved characters with their details."""
        return [
            {
                'id': char_id,
                **char_data
            }
            for char_id, char_data in self.characters.items()
        ]
    
    def delete_character(self, character_id: str) -> bool:
        """Delete a character profile."""
        if character_id in self.characters:
            del self.characters[character_id]
            self._save_characters()
            return True
        return False
    
    def generate_character_prompt(self, character_id: str, scene_description: str) -> str:
        """
        Generate a prompt using a saved character with new scene/pose.
        
        Args:
            character_id: ID of saved character
            scene_description: New scene/pose/clothing (e.g., "sitting in café, white blouse")
        
        Returns:
            Full prompt with character description + scene
        """
        character = self.characters.get(character_id)  # Don't update usage here
        if not character:
            raise ValueError(f"Character '{character_id}' not found")
        
        # Combine fixed character description with variable scene
        return f"{character['description']}, {scene_description}"
