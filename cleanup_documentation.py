#!/usr/bin/env python3
"""
Documentation Management Script
Automatically manages documentation files according to rules
"""
import os
import glob
import shutil
from datetime import datetime

def cleanup_documentation():
    """Clean up documentation files according to rules"""
    
    print("🧹 Starting documentation cleanup...")
    print("=" * 50)
    
    # Categories and their patterns
    categories = {
        'audit': ['*_AUDIT_*.md', '*_audit_*.md'],
        'summary': ['*_SUMMARY.md', '*_summary.md'],
        'guide': ['*_GUIDE.md', '*_INSTRUCTIONS.md', '*_MANUAL.md'],
        'readme': ['README*.md'],
        'telemetry': ['*_TELEMETRY*.md', '*_PHASE*.md']
    }
    
    total_deleted = 0
    
    # Keep only the most recent file in each category
    for category, patterns in categories.items():
        files = []
        for pattern in patterns:
            files.extend(glob.glob(pattern))
        
        if len(files) > 1:
            print(f"\n📁 Category: {category.upper()}")
            print(f"   Found {len(files)} files:")
            
            # Sort by modification time (newest first)
            files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            # Keep the newest, delete the rest
            for i, old_file in enumerate(files[1:], 1):
                try:
                    print(f"   ❌ Deleting: {old_file}")
                    os.remove(old_file)
                    total_deleted += 1
                except Exception as e:
                    print(f"   ⚠️  Error deleting {old_file}: {e}")
            
            print(f"   ✅ Kept: {files[0]}")
        elif len(files) == 1:
            print(f"\n📁 Category: {category.upper()}")
            print(f"   ✅ Single file: {files[0]}")
        else:
            print(f"\n📁 Category: {category.upper()}")
            print(f"   ℹ️  No files found")
    
    print("\n" + "=" * 50)
    print(f"🎉 Cleanup complete! Deleted {total_deleted} old files.")
    
    return total_deleted

def create_documentation_file(category, filename, content):
    """Create a new documentation file, cleaning up old ones first"""
    
    print(f"📝 Creating new {category} file: {filename}")
    
    # Clean up old files in the same category
    cleanup_documentation()
    
    # Create new file
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Successfully created: {filename}")
    except Exception as e:
        print(f"❌ Error creating {filename}: {e}")
        return False
    
    return True

def list_current_documentation():
    """List all current documentation files by category"""
    
    print("📚 Current Documentation Files")
    print("=" * 50)
    
    categories = {
        'audit': ['*_AUDIT_*.md', '*_audit_*.md'],
        'summary': ['*_SUMMARY.md', '*_summary.md'],
        'guide': ['*_GUIDE.md', '*_INSTRUCTIONS.md', '*_MANUAL.md'],
        'readme': ['README*.md'],
        'telemetry': ['*_TELEMETRY*.md', '*_PHASE*.md']
    }
    
    for category, patterns in categories.items():
        files = []
        for pattern in patterns:
            files.extend(glob.glob(pattern))
        
        if files:
            print(f"\n📁 {category.upper()}:")
            for file in files:
                mod_time = datetime.fromtimestamp(os.path.getmtime(file))
                print(f"   📄 {file} (modified: {mod_time.strftime('%Y-%m-%d %H:%M')})")
        else:
            print(f"\n📁 {category.upper()}: No files found")

def backup_important_files():
    """Backup important files before cleanup"""
    
    backup_dir = "backups"
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    important_files = [
        "MISSION_ACCOMPLISHED_SUMMARY.md",
        "ARIA_DEPLOYMENT_INSTRUCTIONS.md",
        "DOCUMENTATION_MANAGEMENT_RULES.md"
    ]
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for file in important_files:
        if os.path.exists(file):
            backup_name = f"{backup_dir}/{file.replace('.md', f'_{timestamp}.md')}"
            try:
                shutil.copy2(file, backup_name)
                print(f"💾 Backed up: {file} → {backup_name}")
            except Exception as e:
                print(f"⚠️  Error backing up {file}: {e}")

def main():
    """Main function with menu options"""
    
    print("📚 Documentation Management Tool")
    print("=" * 50)
    print("1. Clean up documentation files")
    print("2. List current documentation")
    print("3. Backup important files")
    print("4. Full cleanup and backup")
    print("5. Exit")
    
    while True:
        try:
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == '1':
                cleanup_documentation()
            elif choice == '2':
                list_current_documentation()
            elif choice == '3':
                backup_important_files()
            elif choice == '4':
                print("\n🔄 Full cleanup and backup...")
                backup_important_files()
                cleanup_documentation()
                print("\n✅ Full cleanup and backup complete!")
            elif choice == '5':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid option. Please select 1-5.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    # If run directly, do a quick cleanup
    if len(os.sys.argv) > 1 and os.sys.argv[1] == "--quick":
        cleanup_documentation()
    else:
        main()
