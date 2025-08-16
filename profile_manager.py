import json
import os
import time
from datetime import datetime
import logging

class ProfileManager:
    def __init__(self):
        self.profiles_dir = "profiles"
        self.current_profile = "default"
        self.profiles = {}
        self.logger = logging.getLogger(__name__)
        
        # Create profiles directory if it doesn't exist
        if not os.path.exists(self.profiles_dir):
            os.makedirs(self.profiles_dir)
        
        # Load default profiles
        self.load_default_profiles()
        
        # Load custom profiles
        self.load_custom_profiles()
    
    def load_default_profiles(self):
        """Load built-in default profiles"""
        self.profiles = {
            "default": {
                "name": "Default",
                "description": "Balanced settings for general use",
                "settings": {
                    "sensitivity": 1.2,
                    "smoothing": 0.3,
                    "deadzone": 3,
                    "frame_region": 80,
                    "click_threshold": 30,
                    "release_threshold": 40,
                    "drag_threshold": 0.3
                },
                "created": "2024-01-01",
                "tags": ["balanced", "general"]
            },
            "precise": {
                "name": "Precise",
                "description": "High precision for detailed work",
                "settings": {
                    "sensitivity": 0.8,
                    "smoothing": 0.5,
                    "deadzone": 5,
                    "frame_region": 60,
                    "click_threshold": 25,
                    "release_threshold": 35,
                    "drag_threshold": 0.4
                },
                "created": "2024-01-01",
                "tags": ["precise", "detailed", "work"]
            },
            "responsive": {
                "name": "Responsive",
                "description": "Fast response for gaming and quick actions",
                "settings": {
                    "sensitivity": 1.8,
                    "smoothing": 0.2,
                    "deadzone": 2,
                    "frame_region": 100,
                    "click_threshold": 35,
                    "release_threshold": 45,
                    "drag_threshold": 0.2
                },
                "created": "2024-01-01",
                "tags": ["responsive", "gaming", "fast"]
            },
            "stable": {
                "name": "Stable",
                "description": "Reduced jitter for presentations",
                "settings": {
                    "sensitivity": 1.0,
                    "smoothing": 0.7,
                    "deadzone": 8,
                    "frame_region": 90,
                    "click_threshold": 40,
                    "release_threshold": 50,
                    "drag_threshold": 0.5
                },
                "created": "2024-01-01",
                "tags": ["stable", "presentation", "low-jitter"]
            }
        }
    
    def load_custom_profiles(self):
        """Load custom profiles from JSON files"""
        try:
            for filename in os.listdir(self.profiles_dir):
                if filename.endswith('.json'):
                    profile_path = os.path.join(self.profiles_dir, filename)
                    with open(profile_path, 'r') as f:
                        profile_data = json.load(f)
                        profile_name = os.path.splitext(filename)[0]
                        self.profiles[profile_name] = profile_data
                        self.logger.info(f"Loaded custom profile: {profile_name}")
        except Exception as e:
            self.logger.warning(f"Failed to load custom profiles: {e}")
    
    def save_profile(self, name, settings, description="", tags=None):
        """Save a new profile or update existing one"""
        try:
            profile_data = {
                "name": name,
                "description": description,
                "settings": settings,
                "created": datetime.now().strftime("%Y-%m-%d"),
                "modified": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "tags": tags or []
            }
            
            self.profiles[name] = profile_data
            
            # Save to file
            profile_path = os.path.join(self.profiles_dir, f"{name}.json")
            with open(profile_path, 'w') as f:
                json.dump(profile_data, f, indent=2)
            
            self.logger.info(f"Profile '{name}' saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save profile '{name}': {e}")
            return False
    
    def delete_profile(self, name):
        """Delete a profile"""
        try:
            if name in self.profiles:
                # Remove from memory
                del self.profiles[name]
                
                # Remove file
                profile_path = os.path.join(self.profiles_dir, f"{name}.json")
                if os.path.exists(profile_path):
                    os.remove(profile_path)
                
                self.logger.info(f"Profile '{name}' deleted successfully")
                return True
            else:
                self.logger.warning(f"Profile '{name}' not found")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to delete profile '{name}': {e}")
            return False
    
    def get_profile(self, name):
        """Get a specific profile"""
        return self.profiles.get(name)
    
    def list_profiles(self):
        """List all available profiles"""
        return list(self.profiles.keys())
    
    def get_profile_info(self, name):
        """Get detailed information about a profile"""
        profile = self.profiles.get(name)
        if profile:
            return {
                "name": profile["name"],
                "description": profile["description"],
                "created": profile["created"],
                "modified": profile.get("modified", "Never"),
                "tags": profile.get("tags", []),
                "settings_count": len(profile["settings"])
            }
        return None
    
    def search_profiles(self, query):
        """Search profiles by name, description, or tags"""
        results = []
        query_lower = query.lower()
        
        for name, profile in self.profiles.items():
            # Search in name
            if query_lower in name.lower():
                results.append(name)
                continue
            
            # Search in description
            if query_lower in profile.get("description", "").lower():
                results.append(name)
                continue
            
            # Search in tags
            for tag in profile.get("tags", []):
                if query_lower in tag.lower():
                    results.append(name)
                    break
        
        return results
    
    def export_profile(self, name, export_path):
        """Export a profile to a specific location"""
        try:
            profile = self.profiles.get(name)
            if not profile:
                return False
            
            with open(export_path, 'w') as f:
                json.dump(profile, f, indent=2)
            
            self.logger.info(f"Profile '{name}' exported to {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export profile '{name}': {e}")
            return False
    
    def import_profile(self, import_path):
        """Import a profile from a file"""
        try:
            with open(import_path, 'r') as f:
                profile_data = json.load(f)
            
            # Validate profile structure
            if not self.validate_profile(profile_data):
                self.logger.error("Invalid profile format")
                return False
            
            # Extract profile name
            profile_name = profile_data["name"]
            
            # Save imported profile
            return self.save_profile(profile_name, profile_data["settings"], 
                                   profile_data.get("description", ""),
                                   profile_data.get("tags", []))
            
        except Exception as e:
            self.logger.error(f"Failed to import profile: {e}")
            return False
    
    def validate_profile(self, profile_data):
        """Validate profile data structure"""
        required_fields = ["name", "settings"]
        required_settings = ["sensitivity", "smoothing", "deadzone", "frame_region"]
        
        # Check required fields
        for field in required_fields:
            if field not in profile_data:
                return False
        
        # Check required settings
        for setting in required_settings:
            if setting not in profile_data["settings"]:
                return False
        
        return True
    
    def create_profile_from_current(self, name, current_settings, description="", tags=None):
        """Create a profile from current settings"""
        return self.save_profile(name, current_settings, description, tags)
    
    def backup_all_profiles(self, backup_dir):
        """Create a backup of all profiles"""
        try:
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = os.path.join(backup_dir, f"profiles_backup_{timestamp}.json")
            
            with open(backup_file, 'w') as f:
                json.dump(self.profiles, f, indent=2)
            
            self.logger.info(f"All profiles backed up to {backup_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to backup profiles: {e}")
            return False
    
    def restore_profiles_from_backup(self, backup_file):
        """Restore profiles from a backup file"""
        try:
            with open(backup_file, 'r') as f:
                backup_data = json.load(f)
            
            # Clear current profiles
            self.profiles.clear()
            
            # Restore from backup
            for name, profile in backup_data.items():
                if self.validate_profile(profile):
                    self.profiles[name] = profile
                    # Save to individual files
                    profile_path = os.path.join(self.profiles_dir, f"{name}.json")
                    with open(profile_path, 'w') as f:
                        json.dump(profile, f, indent=2)
            
            self.logger.info(f"Profiles restored from {backup_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore profiles: {e}")
            return False
    
    def get_profile_statistics(self):
        """Get statistics about profiles"""
        total_profiles = len(self.profiles)
        custom_profiles = len([p for p in self.profiles.keys() if p not in ["default", "precise", "responsive", "stable"]])
        
        # Count profiles by tag
        tag_counts = {}
        for profile in self.profiles.values():
            for tag in profile.get("tags", []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        return {
            "total_profiles": total_profiles,
            "built_in_profiles": total_profiles - custom_profiles,
            "custom_profiles": custom_profiles,
            "tag_distribution": tag_counts,
            "most_used_tags": sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }

def main():
    """Test the profile manager"""
    pm = ProfileManager()
    
    print("Available profiles:")
    for name in pm.list_profiles():
        info = pm.get_profile_info(name)
        print(f"  {name}: {info['description']}")
    
    print("\nProfile statistics:")
    stats = pm.get_profile_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
