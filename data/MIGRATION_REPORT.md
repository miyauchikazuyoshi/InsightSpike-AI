# Data Structure Migration Report
Generated: 2025-07-21T10:33:44.697380

## Directories to be removed:

### core/
- Files: 1
- Size: 0.0 MB
- Status: Will be removed

### db/
- Files: 2
- Size: 0.1 MB
- Status: Will be removed

### processed/
- Files: 1
- Size: 0.0 MB
- Status: Will be removed

### experiments/
- Files: 0
- Size: 0.0 MB
- Status: Will be removed

### backup/
- Files: 0
- Size: 0.0 MB
- Status: Will be removed

### clean_backup/
- Files: 6
- Size: 0.1 MB
- Status: Will be removed

## New structure created:

```
data/
├── sqlite/
├── knowledge_base/
│   ├── initial/
│   └── samples/
├── models/
├── logs/
└── cache/
```

## Backup location:
data/migration_backup_20250721_103344
