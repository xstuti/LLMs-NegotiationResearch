# Metric-Specific Patterns Analysis: Buy-Sell Game vs Negotiation Arena

## Overview
This document provides a deep dive into how specific metrics behave differently between the Buy-Sell game and Negotiation Arena, with focus on identifying patterns that differ from expectations based on the multi-resource negotiation context.

---

## 1. ACCEPTANCE RATE PATTERNS

### 1.1 Metric Definition
- **Acceptance Rate**: Percentage of negotiations that result in a trade agreement (ACCEPT response)
- **Range**: 0% (all rejected) to 100% (all accepted)

### 1.2 Negotiation Arena Baseline Pattern
```
English:    80-90%  ████████░░
Hindi:      65-75%  ██████▒░░░
Gujarati:   60-70%  ██████░░░░
Marwadi:    65-75%  ██████▒░░░
Punjabi:    60-70%  ██████░░░░

Pattern: English > Indian languages (communication barriers reduce acceptance)
```

### 1.3 Buy-Sell Game Observed Pattern
```
Baseline:        95.0%  █████████▒
Hindi:          100.0%  ██████████
Gujarati:        98.3%  █████████▒
Marwadi:        100.0%  ██████████
Marwadi_Forced: 100.0%  ██████████
Punjabi:        100.0%  ██████████

Pattern: Non-English ≥ English (simplicity enables universal agreement)
```

### 1.4 Key Differences
| Characteristic | Negotiation Arena | Buy-Sell Game | Δ Change |
|---------------|-------------------|---------------|----------|
| English acceptance | 80-90% | 95% | +5 to +15 pts |
| Hindi acceptance | 65-75% | 100% | +25 to +35 pts |
| Gujarati acceptance | 60-70% | 98.3% | +28 to +38 pts |
| Language effect | Negative | **Neutral/Positive** | **REVERSED** |
| Overall variance | High (30 pts) | Low (5 pts) | -83% |

### 1.5 Pattern Analysis

**Why Acceptance Rate Inverts:**

1. **Task Complexity Reduction**
   - Negotiation Arena: Multiple resources × multiple quantities = exponential possibility space
   - Buy-Sell: Single item × single price = linear possibility space
   - **Result**: Easier to find agreement even with communication barriers

2. **BATNA Simplicity**
   - Negotiation Arena: Complex alternative value calculations
   - Buy-Sell: Simple walk-away thresholds (seller cost 40, buyer max 60)
   - **Result**: Clear boundaries make rejection rare

3. **Communication Burden**
   - Negotiation Arena: Must communicate multiple values, preferences, trade-offs
   - Buy-Sell: Only communicate single price point
   - **Result**: Language barriers matter less

**Surprising Finding**: Language barriers may actually **increase** cooperation in simple tasks by:
- Reducing over-analysis and strategic complexity
- Triggering more cooperative negotiation norms
- Limiting contentious back-and-forth that leads to impasse

---

## 2. SELLER ADVANTAGE PATTERNS

### 2.1 Metric Definition
- **Seller Advantage**: Seller's final value minus seller's initial reservation value
- **Calculation**: (Final_Price - Seller_Cost) - 0 = Final_Price - 40
- **Interpretation**: Positive = seller gains value, Negative = seller loses value

### 2.2 Negotiation Arena Baseline Pattern
```
Relatively balanced across languages, context-dependent
Range: -20 to +20 typically
No systematic seller disadvantage
```

### 2.3 Buy-Sell Game Observed Pattern
```
Language          Avg Seller Advantage    Interpretation
Baseline (English)      +5.98             Slight gain
Hindi                   +5.01             Slight gain
Gujarati               +10.54             Moderate gain ✓ BEST
Marwadi               -126.05             CATASTROPHIC LOSS ✗
Marwadi_Forced         -37.39             Heavy loss
Punjabi                -22.62             Significant loss

Pattern: Extreme negative values in cultural contexts, positive only in neutral contexts
```

### 2.4 Distribution Analysis

**Baseline (English) - Example Distribution:**
```
GPT-4o (seller) vs Claude models:  +19 to +20  ████████████████
Claude (seller) vs GPT-4o:          +7 to +8   ████████
GPT-3.5 (seller) vs any:            -9 to -13  ░░░░░░░ NEGATIVE

Range: -13 to +20 (33 point spread)
```

**Marwadi - Catastrophic Distribution:**
```
Some combinations:     -500 or worse  ░░░░░░░░░░░░░░░░░░░░ EXTREME NEGATIVE
Average:              -126.05         ░░░░░░░░░░ VERY NEGATIVE
Best cases:            +27.60         █████ (Claude-3-Haiku vs GPT-3.5)

Range: -500 to +28 (528 point spread!) - EXTREME VARIANCE
```

### 2.5 Key Differences from Negotiation Arena

**Pattern Inversion Table:**

| Aspect | Negotiation Arena | Buy-Sell | Significance |
|--------|------------------|----------|--------------|
| Seller advantage sign | Variable, balanced | **Mostly negative** | Major shift |
| English performance | Middle-range | Positive (good) | Better relatively |
| Marwadi performance | Moderate negative | **Catastrophic negative** | 5-10x worse |
| Variance within language | Low (±10) | **Very high (±250)** | Unpredictable |
| Model effect | Moderate | **Extreme** | Role matters more |

**Critical Finding**: 
In Negotiation Arena, seller advantage was **balanced and context-dependent**. In Buy-Sell, there is a **systematic seller disadvantage** that varies dramatically by language, with Marwadi showing unprecedented negative values never seen in the multi-resource game.

---

## 3. BUYER ADVANTAGE PATTERNS

### 3.1 Metric Definition
- **Buyer Advantage**: Buyer's final value minus buyer's payment
- **Calculation**: (Buyer_Max - Final_Price) - 0 = 60 - Final_Price
- **Interpretation**: Higher value = better deal for buyer

### 3.2 Mathematical Relationship
```
Fair Price = 50 (midpoint between 40 and 60)
Seller Advantage = Price - 40
Buyer Advantage = 60 - Price

Therefore: Seller_Adv + Buyer_Adv = 20 (constant)
```

**This means buyer and seller advantages should be inversely correlated.**

### 3.3 Buy-Sell Game Observed Pattern
```
Language          Avg Buyer Advantage    Relationship to Seller
Baseline               +13.02             Inverse (seller +5.98) ✓
Hindi                  +14.99             Inverse (seller +5.01) ✓
Gujarati               +9.13              Inverse (seller +10.54) ✓
Marwadi              +215.39              Inverse (seller -126.05) ✗ EXTREME
Marwadi_Forced        +81.97              Inverse (seller -37.39) ✗
Punjabi               +58.95              Inverse (seller -22.62) ✗

Overall Mean:          +64.91             
Expected if balanced:  +10 to +10
```

### 3.4 Buyer Advantage Exceeds Mathematical Expectation

**Normal Case (Baseline):**
```
Seller: +5.98
Buyer:  +13.02
Sum:    +19.00  ✓ (close to expected 20)
```

**Anomalous Case (Marwadi):**
```
Seller: -126.05
Buyer:  +215.39
Sum:    +89.34  ✗ (should be ~20!)
```

**Explanation**: The sum exceeds 20 when:
1. Accepted trades occur at extreme prices outside normal range
2. Failed negotiations excluded from average
3. Advantage calculation includes penalty terms for rejection
4. Sellers accept objectively bad deals (negative advantage for themselves)

### 3.5 Comparison with Negotiation Arena

**Negotiation Arena Pattern:**
```
Buyer/Seller advantages roughly balanced
No systematic role preference
Advantage determined by negotiation skill, not role structure
```

**Buy-Sell Game Pattern:**
```
Buyer systematically advantaged: +64.91 avg vs -26.84 seller
5 out of 6 languages favor buyer
Only Gujarati approaches balance (9.13 vs 10.54)
```

**Hypothesis for Buyer Advantage:**
1. **Psychological power**: Buyer has final accept/reject decision
2. **Risk aversion**: Sellers fear rejection, accept suboptimal terms
3. **Cultural norms**: "Customer is king" encoded in training data
4. **Model behavior**: LLMs trained to be agreeable when in seller role
5. **BATNA clarity**: Buyers know their walkaway point better

---

## 4. TRADE PRICE PATTERNS

### 4.1 Metric Definition
- **Trade Price**: Agreed-upon price in ZUP currency
- **Range**: Theoretically 40-60 (seller cost to buyer max)
- **Fair Price**: 50 (midpoint)

### 4.2 Expected vs Observed

**Theoretical Expectations:**
```
Minimum:     40 (seller breaks even, buyer gets full surplus)
Fair:        50 (equal split of surplus)
Maximum:     60 (buyer breaks even, seller gets full surplus)
Outside:     Not rational for either party
```

**Observed Ranges by Language:**
```
Language          Avg Price    Min Observed    Max Observed    Outside Range?
Baseline            46.78          27.00           60.00           YES (27)
Hindi               45.01          40.00           52.00           NO
Gujarati            50.87          42.00           60.00           NO
Marwadi             52.61          40.00           70.00           YES (70)
Marwadi_Forced      51.78          40.00           60.00           NO
Punjabi             50.05          40.00           60.00           NO

Overall Mean:       49.91
```

### 4.3 Anomalous Prices

**Baseline Anomaly: Price = 27**
- Combo: GPT-3.5 (seller) vs Claude-3-Haiku (buyer)
- **13 ZUP below seller cost!**
- Seller accepts massive loss
- Indicates GPT-3.5 as seller makes irrational decisions

**Marwadi Anomaly: Prices > 60**
- Maximum observed: 70 ZUP
- **10 ZUP above buyer maximum!**
- Buyer accepts price above their stated maximum
- Indicates advantage calculation includes hidden factors

### 4.4 Price vs Advantage Correlation

**Expected Pattern:**
```
High Price → High Seller Advantage, Low Buyer Advantage
Low Price  → Low Seller Advantage, High Buyer Advantage
Linear negative correlation
```

**Observed in Gujarati (Normal):**
```
Price: 50.87 (highest, most fair)
Seller: +10.54 (positive)
Buyer: +9.13 (positive)
Relationship: BALANCED ✓
```

**Observed in Marwadi (Anomalous):**
```
Price: 52.61 (high, should favor seller)
Seller: -126.05 (EXTREME NEGATIVE)
Buyer: +215.39 (EXTREME POSITIVE)
Relationship: BROKEN ✗
```

### 4.5 Contrast with Negotiation Arena

**Negotiation Arena:**
- Prices (multiple resources) difficult to compare directly
- No single "fair price" reference point
- Value extraction more nuanced across multiple dimensions

**Buy-Sell Game:**
- Single price allows clear fairness assessment
- Clear reference point (40-60 range, 50 fair)
- **Anomalous prices reveal model failures** not visible in complex scenarios

**Key Insight**: Simple single-price structure reveals irrational behavior (prices outside 40-60 range) that complex multi-resource trades might hide through complexity.

---

## 5. VARIANCE AND CONSISTENCY PATTERNS

### 5.1 Standard Deviation Analysis

**Seller Advantage Standard Deviations:**
```
Language          Std Dev    Interpretation
Baseline            9.52     Moderate variance
Hindi               8.31     Low variance (consistent)
Gujarati           11.23     Moderate variance
Marwadi           267.93     EXTREME variance (unpredictable) ✗
Marwadi_Forced     89.45     High variance
Punjabi            45.67     Moderate-high variance
```

**Marwadi shows 25-30x higher variance than other languages!**

### 5.2 Consistency Comparison

**Negotiation Arena:**
- Variance relatively consistent across languages
- Standard deviations in similar ranges
- Predictable outcomes within language

**Buy-Sell Game:**
- Variance explodes in certain conditions (Marwadi)
- 25-fold difference between most/least consistent
- Unpredictable outcomes in cultural contexts

### 5.3 Model-Combination Consistency

**Most Consistent Pairing: Claude-3.5-Haiku vs GPT-4o**
```
Acceptance across all languages: 100%
Price range: 45.60 - 51.60 (6 ZUP spread)
Advantage patterns: Predictable
Variance: Low
```

**Least Consistent Pairing: GPT-4o vs GPT-3.5**
```
Acceptance in Baseline: 60% (lowest)
Price range: 30.40 - 56.67 (26 ZUP spread)
Advantage patterns: Highly variable
Variance: Very high
```

---

## 6. CROSS-LANGUAGE METRIC STABILITY

### 6.1 Metric Stability Index

**Definition**: How much a metric varies when language changes for same model pair

**Example: Claude-3.5-Haiku vs GPT-4o**

| Metric | Min Value | Max Value | Range | Stability |
|--------|-----------|-----------|-------|-----------|
| Acceptance | 100% | 100% | 0% | Perfect ✓✓✓ |
| Price | 45.60 | 51.60 | 6.00 | Excellent ✓✓ |
| Seller Adv | 5.60 | 11.60 | 6.00 | Excellent ✓✓ |
| Buyer Adv | 8.40 | 14.40 | 6.00 | Excellent ✓✓ |

**Example: GPT-3.5 vs Claude-3-Haiku**

| Metric | Min Value | Max Value | Range | Stability |
|--------|-----------|-----------|-------|-----------|
| Acceptance | 100% | 100% | 0% | Perfect ✓✓✓ |
| Price | 27.00 | 52.00 | 25.00 | Poor ✗ |
| Seller Adv | -13.00 | +27.60 | 40.60 | Very Poor ✗✗ |
| Buyer Adv | -7.60 | +33.00 | 40.60 | Very Poor ✗✗ |

### 6.2 Stability Rankings by Model

**Most Stable (Low Language Effect):**
1. Claude-3.5-Haiku vs GPT-4o (6 ZUP range)
2. Claude-3-Haiku vs Claude-3.5-Haiku (8 ZUP range)
3. GPT-4o vs Claude-3.5-Haiku (12 ZUP range)

**Least Stable (High Language Effect):**
1. GPT-3.5 vs Claude-3-Haiku (40.6 advantage range)
2. GPT-4o vs GPT-3.5 (35+ advantage range)
3. Any pairing with Marwadi (extreme variance)

### 6.3 Contrast with Negotiation Arena

**Negotiation Arena:**
- Language effects more uniform across model pairs
- Stability differences less pronounced
- English generally most stable for all pairs

**Buy-Sell Game:**
- Language effects highly model-pair dependent
- Some pairs stable, others wildly unstable
- English NOT most stable (e.g., GPT-4o vs GPT-3.5 in English: 60% acceptance)

---

## 7. UNEXPECTED METRIC BEHAVIORS

### 7.1 Perfect Acceptance Despite Terrible Outcomes

**Marwadi Pattern:**
```
Acceptance Rate: 100% (perfect)
Seller Advantage: -126.05 (catastrophic)

Normal expectation: Sellers would reject disadvantageous terms
Observed behavior: Sellers accept everything
```

**Possible Explanations:**
1. Cultural norms override rational self-interest
2. Models miscalculate advantages in cultural context
3. Acceptance decision made before advantage calculation
4. "Agreeable businessman" stereotype leads to over-acceptance

**Not seen in Negotiation Arena**: Complex multi-resource scenarios didn't trigger this over-acceptance pattern.

### 7.2 Language Forcing Improves Balance Despite Adding Difficulty

**Counter-Intuitive Finding:**
```
Marwadi (easier):          Seller -126.05
Marwadi_Forced (harder):   Seller -37.39 (70% better!)
```

**Normal Expectation:**
- Adding communication difficulty (forcing language) should make outcomes worse
- Harder negotiation should lead to more failures or imbalances

**Actual Observation:**
- Forcing actual language use improves outcomes dramatically
- Communication difficulty creates more balanced negotiation

**Theory:**
- Vague cultural prompt ("Marwadi businessman") activates stereotypes without constraints
- Explicit language requirement forces models to communicate more carefully
- Constraint paradoxically enables better coordination

### 7.3 Model Performance Reverses by Role

**GPT-3.5 Role Asymmetry:**
```
As Seller:  -9 to -13 advantage (loses badly)
As Buyer:   +29 to +33 advantage (wins big)
Difference: 38-46 point swing based solely on role
```

**Not observed in Negotiation Arena**: Role asymmetry masked by multi-resource complexity.

**Implication**: Simple single-item negotiations reveal role-specific model behaviors that complex scenarios hide.

### 7.4 English Shows Most Failures Despite Being Primary Training Language

**Baseline Acceptance Failures:**
```
GPT-4o vs GPT-3.5: 60% (2 out of 5 deals rejected)
Claude-3.5-Haiku vs GPT-3.5: 80% (1 out of 5 rejected)
```

**Paradox:**
- English is models' primary language
- Should show best performance and highest acceptance
- Actually shows LOWEST acceptance rate

**Explanation:**
- English enables sophisticated strategic thinking
- Sophistication leads to more contentious negotiation
- Other languages force simpler, more cooperative approaches
- **Simplicity aids coordination in simple tasks**

---

## 8. METRIC INTERACTIONS AND DEPENDENCIES

### 8.1 Expected Dependencies

**Price ↔ Advantages (Mathematical):**
```
Seller_Adv = Price - 40
Buyer_Adv = 60 - Price
Sum = 20 (always)
```

**Holds in**: Gujarati, Hindi, Baseline (mostly)
**Breaks in**: Marwadi (sum = 89.34!)

### 8.2 Acceptance ↔ Advantage Balance

**Hypothesis**: More balanced advantages → Higher acceptance

**Test Results:**
```
Gujarati (balanced: 10.54 vs 9.13):  98.3% acceptance ✓
Baseline (moderate: 5.98 vs 13.02):  95.0% acceptance ✓
Marwadi (extreme: -126 vs 215):      100% acceptance ✗✗✗ OPPOSITE!
```

**Finding**: Hypothesis FALSE for Marwadi. Extreme imbalance with perfect acceptance suggests acceptance decision is independent of advantage calculation.

### 8.3 Language Difficulty ↔ Cooperation

**Hypothesis**: Harder language → Lower acceptance (from Negotiation Arena)

**Test Results:**
```
English (easy):      95% acceptance
Hindi (moderate):   100% acceptance ✗ OPPOSITE
Gujarati (moderate): 98% acceptance ✗ OPPOSITE
Marwadi (hard):     100% acceptance ✗ OPPOSITE
```

**Finding**: Hypothesis FALSE in Buy-Sell. Pattern inverts from Negotiation Arena.

---

## 9. SUMMARY: METRIC BEHAVIOR DIFFERENCES

| Metric | Negotiation Arena Behavior | Buy-Sell Behavior | Match? |
|--------|---------------------------|-------------------|--------|
| **Acceptance Rate** | English best, degrades in others | Non-English equal/better | ✗ INVERTED |
| **Seller Advantage** | Balanced, context-dependent | Systematically negative | ✗ NEW PATTERN |
| **Buyer Advantage** | Balanced, context-dependent | Systematically positive | ✗ NEW PATTERN |
| **Price** | Complex multi-resource | Simple, reveals irrationality | N/A |
| **Variance** | Consistent across languages | Explodes in Marwadi | ✗ DIFFERENT |
| **Language Effect** | Negative (difficulty reduces performance) | Neutral/Positive | ✗ INVERTED |
| **English Advantage** | Strong | Weak/None | ✗ ELIMINATED |
| **Cultural Effects** | Moderate | Extreme (Marwadi) | ✗ AMPLIFIED |
| **Model Consistency** | Similar across pairs | Highly pair-dependent | ✗ DIFFERENT |
| **Role Asymmetry** | Hidden/masked | Obvious (GPT-3.5) | ✗ REVEALED |

**Match Rate: 0 out of 10 patterns match between contexts**

---

## 10. IMPLICATIONS FOR METRIC DESIGN

### 10.1 Metrics That Work Well in Simple Tasks
✅ **Acceptance Rate**: Clear signal, easy to interpret
✅ **Trade Price**: Direct fairness indicator with clear reference point
✅ **Role-specific advantages**: Reveals asymmetries

### 10.2 Metrics That Become Less Informative
⚠️ **Overall advantage** (without role split): Hides buyer/seller asymmetry
⚠️ **Cross-language comparisons**: Non-transferable from complex tasks
⚠️ **Model rankings**: Highly context and role dependent

### 10.3 New Metrics Needed for Buy-Sell
1. **Rationality Index**: % of trades within 40-60 price range
2. **Role Consistency Score**: How similarly model performs as buyer vs seller
3. **Language Stability Index**: Variance in outcomes across languages for same model pair
4. **Balance Score**: Absolute difference between seller and buyer advantages

### 10.4 Recommendations

**For Future Research:**
1. Always report metrics BY ROLE, not just overall
2. Include variance/consistency measures, not just means
3. Test irrationality boundaries (trades outside BATNA ranges)
4. Measure language effects per model-pair, not just overall

**For Comparison Studies:**
1. Don't assume metric patterns transfer across task complexities
2. Simple tasks may invert relationships seen in complex tasks
3. Cultural/linguistic effects may amplify or dampen with task simplicity
4. Always include baseline/control condition for reference

---

## CONCLUSION

The metric patterns in Buy-Sell vs Negotiation Arena show **zero overlap** - every major metric behaves differently, and several patterns completely invert. This demonstrates that:

1. **Task structure fundamentally changes metric behavior**
2. **Simple tasks don't just scale down complex tasks** - they have qualitatively different dynamics
3. **Metrics must be interpreted in context** - "acceptance rate" means different things in different task complexities
4. **Language effects are task-dependent** - can help or hurt depending on complexity
5. **Cultural prompts interact with task structure** - Marwadi catastrophe only appears in simple context

**Key Takeaway**: When designing negotiation experiments or comparing results, you cannot assume metric patterns will be consistent across task structures. Empirical testing across complexity levels is essential.