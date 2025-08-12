# RWS Outlier Detection Fix - "≥2 Occurrence = Not Coincidence"

## Issue
RWS outlier detection was incorrectly flagging physiological repeated values as outliers, affecting RWS calculations.

## Root Cause Analysis
Enhanced RWS used `frame_quality_analyzer.py` for outlier detection, which had wrong thresholds:

### **Original Algorithm Problems:**
```python
frequent_threshold = 0.20  # If >20% of frames have same value, consider it normal
...
if frequency_pct > frequent_threshold:  # Wrong: percentage & > instead of >=
```

**Issues:**
1. **Percentage-based**: Required >20% of frames to have same value
2. **Too restrictive**: 10 frames needed >2 occurrences (impossible for 2 occurrences)  
3. **Wrong operator**: Used `>` instead of `>=`
4. **Small tolerance**: 0.02mm too restrictive for floating point precision

## User's Physiological Rule
> **"Aynı değerin iki defa olması tesadüf olamaz"**
> ("Same value appearing twice cannot be coincidence")

### **Cardiac Cycle Logic:**
- **Systole**: Vessel stays at minimum diameter → Repeated min values = NORMAL
- **Diastole**: Vessel stays at maximum diameter → Repeated max values = NORMAL  
- **Single occurrences**: Could be noise/artifacts → Potential outliers

## Solution

### **Fixed Algorithm:**
```python
min_occurrence_threshold = 2  # If ≥2 frames have same value, it's not coincidence - keep it
...
if occurrence_count >= min_occurrence_threshold:
    frequent_values.add(val)
    logger.info(f"→ PROTECTED: Physiological value (≥{min_occurrence_threshold} times - not coincidence)")
```

### **Changes Made:**

#### `/src/analysis/frame_quality_analyzer.py`

**1. Replaced percentage with absolute count (lines 117-131):**
```python
# Before:
frequent_threshold = 0.20  # 20% threshold
if frequency_pct > frequent_threshold:

# After:  
min_occurrence_threshold = 2  # Absolute count
if occurrence_count >= min_occurrence_threshold:
```

**2. Increased tolerance for measurement precision (line 103):**
```python
# Before:
tolerance = 0.02  # 0.02mm tolerance

# After:
tolerance = 0.05  # 0.05mm tolerance (accounts for measurement precision)
```

**3. Updated logging messages:**
- Clear indication of physiological vs noise classification
- Shows absolute counts instead of percentages
- Explains "not coincidence" logic

## Expected Behavior After Fix

### **Extreme Values:**
- **≥2 occurrences** → **PROTECTED** (kept for RWS calculation)
- **1 occurrence** → **Potential outlier** (may be removed)

### **Example Cardiac Cycle:**
```
Frame 1: 2.1mm (systole)  ← Single occurrence, potential outlier
Frame 2: 1.8mm (systole)  ← 
Frame 3: 1.8mm (systole)  ← Same as Frame 2: PROTECTED (≥2 times)
Frame 4: 1.8mm (systole)  ← Same as Frame 2,3: PROTECTED  
Frame 5: 3.2mm (diastole) ←
Frame 6: 3.2mm (diastole) ← Same as Frame 5: PROTECTED (≥2 times)
```

**Result:** Frames 2,3,4,5,6 kept for RWS calculation. Frame 1 might be removed as outlier.

## Benefits

1. **Physiologically Accurate**: Respects cardiac cycle behavior
2. **Preserves Real Data**: Doesn't remove legitimate repeated measurements
3. **Removes Noise**: Still filters single-occurrence artifacts
4. **Better RWS Results**: More accurate min/max MLD detection

## Testing
Test with cardiac cycle data containing repeated MLD values to verify:
- ≥2 identical values are protected
- Single anomalous values are still detected as outliers
- RWS calculations use correct min/max values

## Status: ✅ COMPLETED

RWS outlier detection now follows the physiological rule: **"≥2 occurrence = not coincidence"**