# Cleanup Plan: Remove Legacy and Backup Files

## Files to Remove
1. **Legacy Implementation**: `src/insightspike/legacy/` - 旧実装
2. **Backup Files**: `backup/` - バックアップファイル群
3. **Git Legacy Branches**: Remote legacy branch refs

## Analysis: Legacy Dependencies
- `get_legacy_config()` is used in:
  - `src/insightspike/detection/eureka_spike.py`
  - `src/insightspike/__init__.py` 
  - `src/insightspike/config/__init__.py`
  - `src/insightspike/config.py`

## Migration Strategy
1. ✅ Modern config system is already implemented in `src/insightspike/core/config.py`
2. ⚠️ Need to update EurekaDetector to use modern config directly
3. 🔧 Remove legacy config wrapper after migration

## Safety Check
- Ensure all legacy config usages are replaced with modern equivalents
- Verify no breaking changes in API
- Test EurekaDetector functionality after migration
