# Behavior Update Notes

## Summary

Updated three analysis scripts to properly handle the new social behaviors:
- **Marwadi_Forced**: Forces models to speak only in Marwadi language
- **Baseline**: No social behavior (empty prompts)

## Changes Made

### 1. analyze_real_results.py
- Added `KNOWN_BEHAVIORS` list including "Marwadi_Forced" and "Baseline"
- Added `KNOWN_MODELS` list for better model identification
- Enhanced `parse_directory_name()` method to:
  - Match behavior names against known behaviors list
  - Properly handle underscores in behavior names (e.g., "Marwadi_Forced")
  - Improved fallback parsing logic
  - Better handling of complex model names (e.g., "Claude-3.5-Haiku")

### 2. create_final_heatmaps.py
- Added `KNOWN_BEHAVIORS` constant with new behaviors
- Updated behavior display formatting to replace underscores with spaces
- Behaviors now display as "Marwadi Forced" and "Baseline" in visualizations
- No changes needed to core functionality - works with new behaviors automatically

### 3. generate_final_report.py
- Added `KNOWN_BEHAVIORS` constant with new behaviors
- Updated behavior display formatting throughout the report:
  - Executive summary section
  - Behavior-specific analysis sections
  - Detailed analysis by combination
- Behaviors display as "MARWADI FORCED" and "BASELINE" in report headings
- Underscores replaced with spaces for better readability

## Key Features

### Proper Parsing
The updated parser now correctly handles:
- `GPT-4o_vs_GPT-3.5_Marwadi_Forced_iter_1` ✓
- `Claude-3.5-Haiku_vs_GPT-oss_Marwadi_Forced_iter_3` ✓
- `GPT-4o_vs_Claude-3.5-Haiku_Baseline_iter_2` ✓

### Display Formatting
- Internal storage: `Marwadi_Forced` (with underscore)
- Display in visualizations: "Marwadi Forced" (spaces)
- Display in reports: "MARWADI FORCED" (uppercase with spaces)

## Testing

A test script `test_behavior_parsing.py` was created to verify parsing:
- Tests 12 different directory name patterns
- Covers both new behaviors (Marwadi_Forced, Baseline)
- Tests all old behaviors (Hindi, Gujarati, Marwadi, Punjabi)
- Tests various model combinations
- **All tests passed ✓**

## Usage

The scripts work exactly as before:

```bash
# Analyze results
python analyze_real_results.py .logs/ultimatum_social_behavior_TIMESTAMP

# Create heatmaps
python create_final_heatmaps.py .logs/ultimatum_social_behavior_TIMESTAMP

# Generate report
python generate_final_report.py .logs/ultimatum_social_behavior_TIMESTAMP
```

## Directory Naming Convention

Games should be stored in directories following this pattern:
```
Model1_vs_Model2_Behavior_iter_N
```

Examples:
- `GPT-4o_vs_GPT-3.5_Marwadi_Forced_iter_1`
- `Claude-3.5-Haiku_vs_GPT-oss_Baseline_iter_3`

The parser is robust and handles:
- Models with hyphens and dots (GPT-4o, GPT-3.5, Claude-3.5-Haiku)
- Models with underscores (GPT-oss)
- Behaviors with underscores (Marwadi_Forced)
- Complex combinations of all of the above

## Known Behaviors List

Current supported behaviors:
1. Hindi
2. Gujarati
3. Marwadi
4. **Marwadi_Forced** (NEW)
5. Punjabi
6. **Baseline** (NEW)

To add more behaviors in the future, simply add them to the `KNOWN_BEHAVIORS` list in each script.

## Known Models List

Current supported models:
1. GPT-4o
2. GPT-3.5
3. GPT-oss
4. Claude-3-Haiku
5. Claude-3.5-Haiku
6. Google-2.0-Flash

To add more models, add them to the `KNOWN_MODELS` list in analyze_real_results.py.