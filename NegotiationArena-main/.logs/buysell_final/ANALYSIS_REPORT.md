# Buy-Sell Game: Cross-Language and Cross-Model Analysis Report

## Executive Summary

This report analyzes 321 buy-sell negotiation games across 6 different language/behavioral conditions (Baseline/English, Hindi, Gujarati, Marwadi, Marwadi_Forced, and Punjabi) with 4 different LLM models (GPT-4o, GPT-3.5, Claude-3.5-Haiku, Claude-3-Haiku). The analysis reveals significant language-dependent patterns in negotiation outcomes, with striking differences from traditional Negotiation Arena patterns.

### Key Findings:
- **Near-perfect acceptance rates** (98.9% overall) across all conditions - dramatically higher than Negotiation Arena
- **Strong buyer advantage** across most language conditions, with extreme buyer advantage in Marwadi (-126.05 seller advantage)
- **Language significantly influences outcomes**, with Gujarati showing balanced outcomes while Marwadi shows extreme buyer bias
- **Baseline (English) shows moderate performance**, not necessarily the best or worst
- **Marwadi_Forced improves balance** compared to original Marwadi, but still shows buyer advantage

---

## 1. Cross-Language Variation Analysis

### 1.1 Overall Language Performance

| Language | Acceptance Rate | Avg Seller Advantage | Avg Buyer Advantage | Avg Trade Price |
|----------|----------------|---------------------|--------------------|--------------------|
| **Baseline (English)** | 95.0% | 5.98 | 13.02 | 46.78 |
| **Hindi** | 100.0% | 5.01 | 14.99 | 45.01 |
| **Gujarati** | 98.3% | 10.54 | 9.13 | 50.87 |
| **Marwadi** | 100.0% | **-126.05** | **215.39** | 52.61 |
| **Marwadi_Forced** | 100.0% | -37.39 | 81.97 | 51.78 |
| **Punjabi** | 100.0% | -22.62 | 58.95 | 50.05 |

### 1.2 Key Observations

#### Acceptance Rates
- **Near-universal acceptance**: 5 out of 6 languages show 100% acceptance rates
- **Baseline (English) is the outlier**: Only 95% acceptance, suggesting more critical evaluation or negotiation breakdown in English
- **Stark contrast to Negotiation Arena**: In the multi-resource Negotiation Arena, acceptance rates varied significantly by language and were generally lower

#### Seller vs Buyer Advantage Patterns

**Balanced Conditions:**
- **Gujarati** (10.54 vs 9.13): Most balanced language, slight seller advantage
- **Baseline/English** (5.98 vs 13.02): Relatively balanced, moderate buyer advantage
- **Hindi** (5.01 vs 14.99): Similar to Baseline, moderate buyer advantage

**Buyer-Favored Conditions:**
- **Punjabi** (-22.62 vs 58.95): Strong buyer advantage
- **Marwadi_Forced** (-37.39 vs 81.97): Very strong buyer advantage
- **Marwadi** (-126.05 vs 215.39): **EXTREME buyer advantage** - sellers losing value!

#### Trade Price Analysis
- **Fair price range**: 40-60 (seller cost: 40, buyer max: 60)
- **Gujarati highest**: 50.87 (closest to middle, explaining balanced advantages)
- **Hindi lowest**: 45.01 (closer to seller cost, explaining buyer advantage)
- **Marwadi**: 52.61 - seemingly balanced price but extreme negative seller advantage suggests sellers are accepting unfavorable terms

### 1.3 Comparison with Negotiation Arena

**Major Differences:**

1. **Acceptance Rate Inversion**
   - **Negotiation Arena**: Variable acceptance rates, often lower in non-English languages due to communication barriers
   - **Buy-Sell**: Near-universal acceptance (98.9%), suggesting simpler single-item negotiation is easier to resolve

2. **Language Effect Pattern**
   - **Negotiation Arena**: English typically showed best outcomes, with degradation in other languages
   - **Buy-Sell**: English (Baseline) shows MODERATE performance; Gujarati actually performs better in balance

3. **Buyer Advantage Emergence**
   - **Negotiation Arena**: More balanced or context-dependent advantage patterns
   - **Buy-Sell**: Systematic buyer advantage in most languages (except Gujarati), especially extreme in Marwadi
   - **Hypothesis**: Single-item negotiation structure may psychologically position buyer as "decision-maker"

4. **Marwadi Anomaly**
   - **Negotiation Arena**: Marwadi showed cultural negotiation patterns but not extreme imbalance
   - **Buy-Sell**: Marwadi shows **catastrophic seller disadvantage** (-126.05 average)
   - **Root cause**: Likely cultural/linguistic associations triggering aggressive buyer behavior or passive seller acceptance

---

## 2. Marwadi vs Marwadi_Forced: Impact of Explicit Language Forcing

### 2.1 Comparative Statistics

| Metric | Marwadi | Marwadi_Forced | Change |
|--------|---------|----------------|--------|
| Acceptance Rate | 100% | 100% | No change |
| Seller Advantage | -126.05 | -37.39 | +88.66 (70% improvement) |
| Buyer Advantage | 215.39 | 81.97 | -133.42 (62% reduction) |
| Trade Price | 52.61 | 51.78 | -0.83 |

### 2.2 Analysis

**Significant Improvement:**
- **Forcing language use** (requiring actual Marwadi language vs just "Marwadi businessman" persona) **dramatically reduces buyer advantage**
- **Seller disadvantage reduced by 70%** - from catastrophic to merely strong
- **Price remains similar**, suggesting the improvement comes from better negotiation dynamics, not price changes

**Remaining Issues:**
- **Still strong buyer advantage** (-37.39 seller advantage) compared to balanced languages
- **Pattern persists**: Language forcing helps but doesn't eliminate the fundamental issue

**Hypothesis:**
- **Original Marwadi prompt** ("You are a Marwadi businessman") may trigger stereotypes about aggressive negotiation without actual language barrier
- **Forced Marwadi** creates genuine communication challenges that paradoxically lead to more balanced outcomes
- **Communication difficulty may equalize power dynamics** by making both parties work harder to reach agreement

### 2.3 Sample Model Comparisons

**Claude-3-Haiku vs GPT-3.5:**
- Marwadi: Seller +27.60, Buyer -7.60 (seller actually wins!)
- Marwadi_Forced: Seller +22.00, Buyer -2.00 (seller still wins, slightly less)

**Claude-3.5-Haiku vs Claude-3-Haiku:**
- Marwadi: Seller +13.00, Buyer +7.00 (balanced)
- Marwadi_Forced: Seller +5.25, Buyer +14.75 (flipped to buyer advantage)

**Observation**: The impact of language forcing is **model-combination dependent**, suggesting different models respond differently to cultural vs linguistic prompts.

---

## 3. Baseline (English) Performance and Patterns

### 3.1 English vs Other Languages

**Baseline Performance:**
- Acceptance Rate: **95%** (lowest of all languages)
- Seller Advantage: 5.98 (3rd best)
- Buyer Advantage: 13.02 (3rd best)
- Trade Price: 46.78 (2nd lowest)

**Key Finding: English is NOT the best performer**
- This contradicts typical LLM behavior where English shows superior performance
- **Gujarati outperforms English** in balance (10.54 vs 9.13 advantage split vs 5.98 vs 13.02)
- **English shows most negotiation failures** (5% rejection rate vs 0-2% for others)

### 3.2 Extreme Cases in Baseline

**High Seller Advantage (Seller Winning):**
- GPT-4o vs Claude-3-Haiku: Seller +20.00, Buyer 0.00, Price 60.00
- GPT-4o vs Claude-3.5-Haiku: Seller +19.00, Buyer +1.00, Price 59.00
- **Pattern**: GPT-4o as seller dominates weaker Claude models

**High Buyer Advantage (Buyer Winning):**
- GPT-3.5 vs Claude-3-Haiku: Seller -13.00, Buyer +33.00, Price 27.00
- GPT-3.5 vs Claude-3.5-Haiku: Seller -9.00, Buyer +29.00, Price 31.00
- GPT-3.5 vs GPT-4o: Seller -9.60, Buyer +29.60, Price 30.40
- **Pattern**: GPT-3.5 as seller loses badly across all opponents

**Low Acceptance:**
- GPT-4o vs GPT-3.5: 60% acceptance (lowest in entire dataset)
- Claude-3.5-Haiku vs GPT-3.5: 80% acceptance
- **Pattern**: GPT-3.5 as buyer rejects more deals, possibly overvaluing

### 3.3 Differences from Negotiation Arena

**Negotiation Arena English Patterns:**
- Generally showed best acceptance rates
- More balanced outcomes
- Served as performance ceiling

**Buy-Sell English Patterns:**
- **Lower acceptance** than other languages
- **Not the most balanced** (Gujarati is)
- **Shows strong model-dependent variance**

**Explanation:**
1. **Simplicity factor**: Single-item negotiation removes complexity advantage of English
2. **Cultural context**: Other languages may bring cultural negotiation norms that help in simple transactions
3. **Model training**: LLMs may have more buy-sell negotiation training data in English, leading to more sophisticated (and thus more contentious) strategies

---

## 4. Model-Specific Analysis

### 4.1 Cross-Language Consistency: Claude-3.5-Haiku vs GPT-4o

| Language | Acceptance | Seller Adv | Buyer Adv | Price |
|----------|------------|------------|-----------|-------|
| Baseline | 100.0% | 7.00 | 13.00 | 47.00 |
| Hindi | 100.0% | 8.60 | 11.40 | 48.60 |
| Gujarati | 100.0% | 11.60 | 8.40 | 51.60 |
| Marwadi | 100.0% | 5.60 | 14.40 | 45.60 |
| Marwadi_Forced | 100.0% | 8.60 | 11.40 | 48.60 |
| Punjabi | 100.0% | 8.00 | 12.00 | 48.00 |

**Observations:**
- **100% acceptance** across ALL languages for this pairing
- **Gujarati shows seller advantage** (11.60 vs 8.40) - unique pattern
- **Marwadi and Marwadi_Forced differ significantly** (5.60 vs 8.60 seller advantage)
- **Hindi and Marwadi_Forced identical outcomes** (8.60 vs 11.40) - interesting convergence

### 4.2 Model Hierarchy

**Strongest Sellers (Average Advantage):**
1. GPT-4o: Consistently high seller advantage when in seller role
2. Claude-3-Haiku: Strong but variable
3. Claude-3.5-Haiku: Moderate
4. GPT-3.5: Weakest seller (often negative advantage)

**Strongest Buyers:**
1. GPT-3.5: Aggressive buyer, often refuses deals
2. GPT-4o: Strong negotiator from either side
3. Claude models: More balanced

**Negotiation Success:**
- Claude-to-Claude pairings: High acceptance, balanced
- GPT-4o vs Claude: High acceptance, GPT-4o advantage
- GPT-3.5 involved: Lower acceptance, high variance

### 4.3 Contrast with Negotiation Arena Model Performance

**Negotiation Arena Patterns:**
- GPT-4o generally strongest in complex multi-resource scenarios
- Language degradation more uniform across models
- Claude models competitive in English, weaker in other languages

**Buy-Sell Patterns:**
- **GPT-4o dominance even stronger** in simple single-item case
- **Claude models perform better relatively** in non-English languages
- **GPT-3.5 shows extreme variance** - excellent buyer, terrible seller

**Explanation**: Simple buy-sell structure may favor GPT-4o's decision-making strengths while complex multi-resource scenarios allow other models to leverage different strategic thinking.

---

## 5. Notable Patterns and Anomalies

### 5.1 The Marwadi Mystery

**Catastrophic Seller Disadvantage in Marwadi:**
- Average seller advantage: **-126.05** (extreme negative)
- Some individual games show seller advantages of -500 or worse
- Trade prices appear reasonable (52.61 average)

**Possible Explanations:**
1. **Cultural stereotype activation**: "Marwadi businessman" may trigger aggressive negotiation expectations
2. **Valuation confusion**: Models may miscalculate advantages in cultural context
3. **Communication breakdown**: Cultural context without actual language use creates uncertainty
4. **Acceptance pressure**: Cultural norms may push sellers to accept disadvantageous terms

**Evidence for Cultural Stereotype Theory:**
- Marwadi_Forced (actual language) performs much better
- Pattern not seen in Negotiation Arena with same cultural prompt
- Single-item context may amplify stereotype effects

### 5.2 Perfect Acceptance in Non-English Languages

**100% acceptance in Hindi, Gujarati, Marwadi, Marwadi_Forced, and Punjabi**

**Contrast with Baseline:**
- English: 95% acceptance
- Non-English: 97-100% acceptance

**Hypothesis:**
1. **Linguistic uncertainty increases cooperation**: When communication is harder, both parties compromise more
2. **Cultural norms**: Non-English prompts may activate cultural cooperation norms
3. **Simplicity of task**: Single-item negotiation easier to resolve even with language barriers
4. **Model behavior**: Models may be more risk-averse when operating in non-primary languages

### 5.3 Price vs Advantage Discrepancy

**Expected relationship**: Higher price → Higher seller advantage, Lower buyer advantage

**Observed in most cases**, but exceptions:

**Marwadi anomaly:**
- Price: 52.61 (5th out of 6, middle-high)
- Seller advantage: -126.05 (catastrophic)
- **Interpretation**: Sellers accepting unfavorable terms at seemingly fair prices

**Gujarati consistency:**
- Price: 50.87 (highest, most fair)
- Advantages: 10.54 vs 9.13 (most balanced)
- **Interpretation**: True middle-ground negotiation

---

## 6. Implications and Recommendations

### 6.1 For Game Design

**Buy-Sell Game Characteristics:**
1. **Extremely high acceptance rates** suggest game may be too simple or cooperative
2. **Language effects are dramatic** but unpredictable (Marwadi case)
3. **Single-item structure** removes complexity that might showcase LLM capabilities
4. **Consider adding**:
   - Multiple rounds of counter-offers
   - Information asymmetry
   - Time pressure or outside options
   - Multiple items or bundles

### 6.2 For Language/Cultural Prompting

**Key Learnings:**
1. **Cultural stereotypes can backfire** (Marwadi example)
2. **Explicit language forcing** (Marwadi_Forced) creates better balance
3. **English is not always optimal** for LLM negotiation tasks
4. **Gujarati prompts** show promise for balanced outcomes
5. **Recommendation**: Use explicit language requirements rather than cultural personas

### 6.3 For Model Selection

**Model Recommendations by Use Case:**

**Balanced Negotiation:**
- Claude-3.5-Haiku vs Claude-3-Haiku (most consistent across languages)

**Testing Extreme Scenarios:**
- GPT-4o vs GPT-3.5 (highest variance, lowest acceptance)

**Cross-Cultural Testing:**
- Use Gujarati or Hindi (balanced, high acceptance)
- Avoid Marwadi without language forcing (extreme imbalance)

### 6.4 Comparison with Negotiation Arena Insights

**What Transfers:**
- Language matters significantly
- Model selection critical for outcomes
- Cultural context affects negotiation

**What Differs:**
- **Acceptance patterns inverted** (simple game = higher acceptance)
- **English advantage disappears** (simplicity removes complexity advantage)
- **Buyer advantage emerges** (not seen in multi-resource scenario)

**Key Insight**: Game complexity interacts with language/cultural effects. Simple games may show different patterns than complex negotiations.

---

## 7. Future Research Directions

### 7.1 Investigate Marwadi Anomaly
- Test with explicit cultural training
- Compare "Marwadi businessman" vs "Marwadi language" vs "Indian businessman"
- Analyze conversation transcripts for stereotype activation

### 7.2 Replicate with More Languages
- Test Romance languages (Spanish, French)
- Test Asian languages (Chinese, Japanese)
- Test with explicit cultural vs linguistic prompts

### 7.3 Complexity Gradient
- Add multi-round negotiations
- Introduce information asymmetry
- Test with multiple items
- Compare patterns to Negotiation Arena

### 7.4 Model Behavior Analysis
- Why does GPT-3.5 excel as buyer but fail as seller?
- Why do Claude models perform better in non-English relatively?
- How do models represent cultural knowledge?

---

## 8. Conclusion

The buy-sell game reveals fascinating patterns that both align with and diverge from Negotiation Arena results:

**Similarities:**
- Language significantly affects negotiation outcomes
- Model selection matters greatly
- Cultural context influences behavior

**Differences:**
- **Higher acceptance rates** in simpler single-item context
- **English loses its advantage** - Gujarati actually performs better
- **Strong buyer advantage** emerges across most languages
- **Cultural stereotypes** (Marwadi) can create extreme imbalances

**Most Striking Finding:**
The **Marwadi paradox** - where cultural prompting without language forcing creates catastrophic seller disadvantage (-126.05 average), while adding explicit language requirements (Marwadi_Forced) improves outcomes by 70%. This suggests cultural stereotypes can backfire in ways not seen in more complex negotiation scenarios.

**Practical Takeaway:**
For LLM-based negotiation systems:
1. Don't assume English is optimal
2. Use explicit language requirements over cultural personas
3. Simple negotiation structures may hide rather than reveal LLM capabilities
4. Test thoroughly across languages - patterns are not always intuitive

The buy-sell results suggest that game structure, language choice, and cultural framing interact in complex ways that require careful empirical testing rather than assumptions based on English-language performance.