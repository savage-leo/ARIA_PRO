# 📚 Documentation Management Rules - ARIA PRO

## 🎯 **PURPOSE**
This document establishes rules for managing documentation files to maintain a clean, organized codebase and prevent documentation bloat.

---

## 📋 **DOCUMENTATION CATEGORIES**

### **1. AUDIT DOCUMENTATION**
- **Purpose**: System audits, line-by-line analysis, comprehensive reviews
- **Naming**: `*_AUDIT_*.md`, `*_audit_*.md`
- **Rule**: Only ONE audit file per category at a time
- **Examples**: 
  - `SYSTEM_AUDIT_REPORT.md`
  - `LINE_BY_LINE_AUDIT_REPORT.md`
  - `DEPLOYMENT_AUDIT_REPORT.md`

### **2. SUMMARY DOCUMENTATION**
- **Purpose**: Executive summaries, status reports, completion summaries
- **Naming**: `*_SUMMARY.md`, `*_summary.md`
- **Rule**: Only ONE summary file per category at a time
- **Examples**:
  - `MISSION_ACCOMPLISHED_SUMMARY.md`
  - `DEPLOYMENT_SUMMARY.md`
  - `PROJECT_SUMMARY.md`

### **3. USAGE GUIDES**
- **Purpose**: How-to guides, integration instructions, user manuals
- **Naming**: `*_GUIDE.md`, `*_INSTRUCTIONS.md`, `*_MANUAL.md`
- **Rule**: Only ONE guide per topic at a time
- **Examples**:
  - `ARIA_DEPLOYMENT_INSTRUCTIONS.md`
  - `INTEGRATION_GUIDE.md`
  - `USER_MANUAL.md`

### **4. README DOCUMENTATION**
- **Purpose**: Project overview, setup instructions, main documentation
- **Naming**: `README*.md`
- **Rule**: Only ONE README per project/component
- **Examples**:
  - `README.md`
  - `README_ARIA_MT5_PRODUCTION.md`
  - `README_LIVE_EXECUTION.md`

### **5. TELEMETRY DOCUMENTATION**
- **Purpose**: Telemetry reports, phase completion reports
- **Naming**: `*_TELEMETRY*.md`, `*_PHASE*.md`
- **Rule**: Only ONE telemetry report per phase
- **Examples**:
  - `PHASE_1_TELEMETRY_COMPLETE.md`
  - `PHASE_2_TELEMETRY_COMPLETE.md`
  - `TELEMETRY_AUDIT_SUMMARY.md`

---

## 🗂️ **FILE MANAGEMENT RULES**

### **RULE 1: ONE FILE PER CATEGORY**
- **Before creating**: Check if a file of the same category exists
- **If exists**: Replace content, don't create new file
- **If new category**: Create new file with appropriate naming

### **RULE 2: TIMESTAMPED FILES**
- **JSON logs**: Keep timestamped (e.g., `system_audit_20250821_082443.json`)
- **Markdown docs**: Replace content, don't timestamp
- **Exception**: Historical snapshots (archive in `/backups/`)

### **RULE 3: CLEANUP BEFORE CREATION**
```bash
# Before creating new audit file:
# 1. Check for existing audit files
# 2. Delete old audit files in same category
# 3. Create new audit file

# Before creating new summary file:
# 1. Check for existing summary files
# 2. Delete old summary files in same category
# 3. Create new summary file
```

### **RULE 4: ARCHIVAL STRATEGY**
- **Keep**: Current working documentation
- **Archive**: Historical snapshots in `/backups/` directory
- **Delete**: Outdated, duplicate, or obsolete files

---

## 🔧 **IMPLEMENTATION SCRIPT**

### **Documentation Cleanup Script**
```python
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
    
    # Categories and their patterns
    categories = {
        'audit': ['*_AUDIT_*.md', '*_audit_*.md'],
        'summary': ['*_SUMMARY.md', '*_summary.md'],
        'guide': ['*_GUIDE.md', '*_INSTRUCTIONS.md', '*_MANUAL.md'],
        'readme': ['README*.md'],
        'telemetry': ['*_TELEMETRY*.md', '*_PHASE*.md']
    }
    
    # Keep only the most recent file in each category
    for category, patterns in categories.items():
        files = []
        for pattern in patterns:
            files.extend(glob.glob(pattern))
        
        if len(files) > 1:
            # Sort by modification time (newest first)
            files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            # Keep the newest, delete the rest
            for old_file in files[1:]:
                print(f"Deleting old {category} file: {old_file}")
                os.remove(old_file)

def create_documentation_file(category, filename, content):
    """Create a new documentation file, cleaning up old ones first"""
    
    # Clean up old files in the same category
    cleanup_documentation()
    
    # Create new file
    with open(filename, 'w') as f:
        f.write(content)
    
    print(f"Created new {category} file: {filename}")

# Usage example:
# create_documentation_file('audit', 'SYSTEM_AUDIT_REPORT.md', content)
```

---

## 📝 **WORKFLOW EXAMPLES**

### **Creating a New Audit Report**
```bash
# 1. Check for existing audit files
ls *_AUDIT_*.md

# 2. Delete old audit files (if any)
rm OLD_AUDIT_REPORT.md

# 3. Create new audit file
# (content will be added via edit_file tool)
```

### **Creating a New Summary**
```bash
# 1. Check for existing summary files
ls *_SUMMARY.md

# 2. Delete old summary files (if any)
rm OLD_SUMMARY.md

# 3. Create new summary file
# (content will be added via edit_file tool)
```

### **Updating Existing Documentation**
```bash
# 1. Use search_replace to update existing file
# 2. Don't create new file with similar name
# 3. Keep the existing file structure
```

---

## 🚫 **WHAT NOT TO DO**

### **❌ DON'T:**
- Create multiple files with similar names
- Keep old versions alongside new versions
- Use timestamps in markdown documentation names
- Create files without checking for existing ones
- Leave orphaned documentation files

### **✅ DO:**
- Replace content in existing files
- Use clear, descriptive names
- Follow the naming conventions
- Clean up before creating new files
- Archive important historical versions

---

## 📊 **CURRENT DOCUMENTATION STATUS**

### **Active Documentation Files:**
- `MISSION_ACCOMPLISHED_SUMMARY.md` - Final project summary
- `ARIA_DEPLOYMENT_INSTRUCTIONS.md` - Deployment guide
- `DOCUMENTATION_MANAGEMENT_RULES.md` - This file

### **Archived Documentation:**
- All old/duplicate files have been cleaned up
- Historical snapshots can be found in `/backups/` (if needed)

---

## 🎯 **ENFORCEMENT**

### **Before Creating New Documentation:**
1. **Check**: Look for existing files in the same category
2. **Clean**: Remove old files in the same category
3. **Create**: Add new content to appropriate file
4. **Verify**: Ensure only one file per category exists

### **Regular Maintenance:**
- Run cleanup script monthly
- Review documentation quarterly
- Archive important historical versions
- Delete obsolete files

---

## 📞 **SUPPORT**

### **For Questions:**
- Check this document first
- Follow the established patterns
- Use the cleanup script when in doubt
- Archive rather than delete if unsure

**Remember**: Clean documentation = Better maintainability = Faster development
