# Buy-Sell vs Negotiation Arena: Key Findings Summary

## Overview
This document summarizes the most important findings from comparing the Buy-Sell game (single-item negotiation) with the Negotiation Arena (multi-resource negotiation) across multiple languages and LLM models.

---

## 1. MAJOR DIFFERENCES FROM NEGOTIATION ARENA

### 1.1 Acceptance Rate Pattern (INVERTED)

| Scenario | Negotiation Arena | Buy-Sell Game |
|----------|-------------------|---------------|
| English/Baseline | High acceptance (~80-90%) | Moderate (95%) |
| Hindi | Lower acceptance (~60-70%) | Perfect (100%) |
| Gujarati | Lower acceptance (~60-70%) | Near-perfect (98.3%) |
| Marwadi | Variable (~65-75%) | Perfect (100%) |
| Punjabi | Lower acceptance (~60-70%) | Perfect (100%) |
| **Pattern** | **English best** | **Non-English equal or better** |

**Key Insight**: Simpler single-item negotiation removes the complexity advantage that English provides in multi-resource scenarios. Language barriers may even increase cooperation.

### 1.2 Language Performance Hierarchy (REVERSED)

**Negotiation Arena:**
```
English > Hindi > Gujarati ≈ Punjabi > Marwadi
(Clarity and model training favor English)
```

**Buy-Sell Game:**
```
Gujarati > Baseline/English ≈ Hindi > Punjabi > Marwadi_Forced > Marwadi
(Simplicity negates English advantage; cultural norms help)
```

**Key Insight**: English loses its "best performer" status in simple negotiations. Gujarati actually shows the most balanced outcomes.

### 1.3 Buyer vs Seller Advantage (NEW PATTERN)

**Negotiation Arena:**
- More balanced or context-dependent
- Role advantage varies by language and model
- No systematic bias toward one role

**Buy-Sell Game:**
- **Strong systematic buyer advantage** (64.91 avg buyer advantage vs -26.84 seller)
- Buyer wins in 5 out of 6 languages
- Only Gujarati shows near-balance (10.54 seller vs 9.13 buyer)

**Hypothesis**: Single-item structure psychologically positions buyer as "decision-maker" with power to accept/reject, creating inherent advantage.

---

## 2. THE MARWADI PARADOX (UNIQUE TO BUY-SELL)

### 2.1 Catastrophic Failure

**Negotiation Arena - Marwadi:**
- Moderate challenges, lower acceptance rates
- Cultural patterns observable but balanced
- Performance degradation similar to other Indian languages

**Buy-Sell Game - Marwadi:**
- **CATASTROPHIC seller disadvantage**: -126.05 average (extreme negative)
- Perfect acceptance (100%) despite terrible outcomes for sellers
- Buyer advantage of +215.39 (extreme positive)
- Pattern NOT seen in Negotiation Arena

### 2.2 Language Forcing Solution

**Original Marwadi Prompt:**
- "You are a Marwadi businessman. Negotiate accordingly."
- Triggers cultural stereotypes without actual language barrier
- Result: Sellers accept highly disadvantageous terms

**Marwadi_Forced Prompt:**
- "You speak and negotiate only in Marwadi language"
- Forces actual linguistic communication
- Result: 70% improvement in seller disadvantage (-126.05 → -37.39)

**Key Insight**: Cultural personas can backfire in simple negotiations more than complex ones. Explicit language requirements perform better than cultural role-playing.

---

## 3. MODEL PERFORMANCE VARIATIONS

### 3.1 Model Rankings Shift by Context

| Model | Negotiation Arena Rank | Buy-Sell Game Rank | Notes |
|-------|----------------------|-------------------|-------|
| **GPT-4o** | 1st (Strong overall) | 1st (Even stronger) | Dominance amplified in simple context |
| **Claude-3.5-Haiku** | 2nd (Competitive) | 2nd-3rd (Variable) | Better relatively in non-English |
| **Claude-3-Haiku** | 3rd (Solid) | 2nd-3rd (Variable) | Surprisingly strong in some languages |
| **GPT-3.5** | 4th (Weaker) | **Paradoxical** | Excellent buyer, terrible seller |

### 3.2 GPT-3.5 Role Paradox (UNIQUE PATTERN)

**As Seller:**
- GPT-3.5 vs Claude-3-Haiku: Seller -13.00, Price 27.00 (loses badly)
- GPT-3.5 vs Claude-3.5-Haiku: Seller -9.00, Price 31.00 (loses badly)
- GPT-3.5 vs GPT-4o: Seller -9.60, Price 30.40 (loses badly)

**As Buyer:**
- High rejection rate (GPT-4o vs GPT-3.5: only 60% acceptance)
- Forces very low prices when accepts
- Aggressive negotiation stance

**Not observed in Negotiation Arena**: Multi-resource complexity masked this role-dependent performance asymmetry.

### 3.3 Claude Models Relatively Stronger in Non-English

**Negotiation Arena:**
- Claude models struggled more in non-English languages
- GPT models maintained performance better

**Buy-Sell Game:**
- Claude models perform relatively better in Gujarati, Hindi, Punjabi
- Performance gap with GPT models narrows
- More consistent acceptance rates

**Hypothesis**: Simpler task structure allows Claude models to leverage different strengths that complex negotiations don't reveal.

---

## 4. LANGUAGE-SPECIFIC INSIGHTS

### 4.1 Gujarati: The Unexpected Champion

**Why Gujarati Outperforms:**
- Most balanced outcomes (10.54 seller vs 9.13 buyer)
- Near-perfect acceptance (98.3%)
- Highest average trade price (50.87 - closest to fair midpoint of 50)
- Consistent across model combinations

**Theory**: Gujarati business culture norms encoded in training data may provide good negotiation heuristics for simple transactions.

### 4.2 Hindi: Similar to Baseline but Better Acceptance

**Similarities to Baseline:**
- Comparable advantage distribution (5.01 vs 14.99 for Hindi; 5.98 vs 13.02 for Baseline)
- Similar price levels (45.01 vs 46.78)

**Key Difference:**
- Hindi: 100% acceptance
- Baseline: 95% acceptance

**Insight**: Hindi creates cooperation without changing fundamental negotiation balance.

### 4.3 Punjabi: Moderate Buyer Advantage

**Pattern:**
- Consistent buyer advantage (-22.62 seller vs 58.95 buyer)
- Perfect acceptance (100%)
- Mid-range pricing (50.05)

**Comparison to Negotiation Arena:**
- Negotiation Arena: Punjabi similar to other Indian languages
- Buy-Sell: Punjabi shows distinct pattern between balanced (Gujarati) and extreme (Marwadi)

---

## 5. BASELINE (ENGLISH) PERFORMANCE ANALYSIS

### 5.1 English is NOT Optimal in Buy-Sell

**Metrics where English underperforms:**
- **Lowest acceptance rate** (95% vs 97-100% for others)
- **Less balanced** than Gujarati (5.98 vs 13.02 split vs 10.54 vs 9.13)
- **Higher variance** in outcomes across model combinations

**Metrics where English performs well:**
- **Lower prices** (46.78 avg - good for buyers)
- **Moderate advantages** (not extreme like Marwadi)

### 5.2 Model Interactions Most Visible in English

**Extreme cases in Baseline (not seen as strongly in other languages):**

**GPT-4o Dominance:**
- GPT-4o vs Claude-3-Haiku: Seller +20, Buyer 0 (complete domination)
- GPT-4o vs Claude-3.5-Haiku: Seller +19, Buyer +1 (near-complete domination)

**GPT-3.5 Weakness:**
- GPT-3.5 vs Claude-3-Haiku: Seller -13, Buyer +33 (massive loss)
- GPT-3.5 vs GPT-4o: 60% acceptance (lowest in dataset)

**Theory**: English allows models to fully express strategic capabilities, revealing performance gaps. Other languages may "level the playing field" through communication constraints.

---

## 6. CRITICAL INSIGHTS FOR RESEARCH & DESIGN

### 6.1 Task Complexity Matters More Than Expected

| Complexity Level | English Advantage | Acceptance Rate | Strategic Depth |
|-----------------|-------------------|-----------------|-----------------|
| **High** (Negotiation Arena) | Strong | Variable (60-90%) | High variance |
| **Low** (Buy-Sell) | Weak/None | Very High (95-100%) | Lower variance |

**Implication**: Don't generalize from one task complexity to another. Test across complexity spectrum.

### 6.2 Cultural Prompts Can Backfire

**Lesson from Marwadi:**
- Vague cultural persona ("Marwadi businessman") → Catastrophic outcomes
- Explicit language requirement ("speak only in Marwadi") → Much better (though still imperfect)

**Best Practice:**
```
❌ BAD:  "You are a [Culture] businessman"
✅ GOOD: "You speak and negotiate only in [Language]"
```

### 6.3 Simple Tasks May Hide Model Capabilities

**Buy-Sell Game:**
- Very high acceptance rates (98.9%) suggest low challenge
- Less strategic variety than Negotiation Arena
- Harder to differentiate model capabilities

**Negotiation Arena:**
- Variable acceptance (60-90%) suggests appropriate challenge
- Multiple resources create strategic complexity
- Better reveals model reasoning capabilities

**Recommendation**: Use task complexity appropriate to research questions. Simple tasks for basic competence, complex tasks for strategic reasoning.

---

## 7. SURPRISING FINDINGS

### 🔴 Finding #1: Perfect Acceptance in "Difficult" Languages
- **Expected**: Lower acceptance in non-English due to communication barriers
- **Observed**: 100% acceptance in Hindi, Gujarati, Marwadi, Punjabi
- **Contrast**: Negotiation Arena showed opposite pattern

### 🔴 Finding #2: English Loses Advantage
- **Expected**: English best performance (as in Negotiation Arena)
- **Observed**: Gujarati more balanced, English has most failures
- **Contrast**: Complete reversal of Negotiation Arena hierarchy

### 🔴 Finding #3: Marwadi Catastrophe
- **Expected**: Moderate performance degradation (as in Negotiation Arena)
- **Observed**: Catastrophic seller disadvantage (-126.05)
- **Contrast**: Magnitude and direction completely different from Negotiation Arena

### 🔴 Finding #4: Systematic Buyer Advantage
- **Expected**: Balanced or random role advantages (as in Negotiation Arena)
- **Observed**: Consistent buyer advantage across most languages
- **Contrast**: New structural pattern not seen in multi-resource game

### 🔴 Finding #5: GPT-3.5 Role Asymmetry
- **Expected**: Consistent model performance regardless of role
- **Observed**: Excellent buyer, terrible seller
- **Contrast**: Not visible in Negotiation Arena's complex scenarios

---

## 8. ACTIONABLE RECOMMENDATIONS

### For Researchers:
1. ✅ **Don't assume English is optimal** - test multiple languages
2. ✅ **Test across task complexity levels** - patterns may reverse
3. ✅ **Use explicit language requirements** over cultural personas
4. ✅ **Consider structural role advantages** in task design
5. ✅ **Analyze by role** (buyer/seller) not just overall performance

### For Practitioners:
1. ✅ **Gujarati prompts** show promise for balanced negotiations
2. ✅ **Avoid vague cultural prompts** (especially Marwadi)
3. ✅ **GPT-4o** consistently strong across contexts
4. ✅ **GPT-3.5** only suitable for buyer role in simple negotiations
5. ✅ **Claude models** good for cross-language robustness

### For Game Designers:
1. ✅ **Single-item negotiation may be too simple** (98.9% acceptance)
2. ✅ **Add complexity** to reveal model capabilities (multi-round, asymmetric info)
3. ✅ **Consider structural role balance** (buyer currently has inherent advantage)
4. ✅ **Test language effects** - they're more dramatic than expected

---

## 9. SUMMARY TABLE: NEGOTIATION ARENA vs BUY-SELL

| Aspect | Negotiation Arena | Buy-Sell Game | Winner |
|--------|------------------|---------------|--------|
| **Acceptance Rates** | Variable (60-90%) | Very High (95-100%) | Arena (more revealing) |
| **English Advantage** | Strong | Weak/None | Arena (shows capability) |
| **Language Effect Size** | Moderate | Extreme (esp. Marwadi) | Buy-Sell (more dramatic) |
| **Role Balance** | Balanced | Buyer favored | Arena (fairer) |
| **Model Differentiation** | Clear hierarchy | More role-dependent | Arena (clearer signal) |
| **Strategic Depth** | High | Low | Arena (more complex) |
| **Cultural Sensitivity** | Moderate | Extreme (Marwadi paradox) | Buy-Sell (shows risk) |
| **Best Language** | English | Gujarati | Different! |
| **Research Value** | High (complex reasoning) | Medium (basic competence) | Arena (richer data) |
| **Practical Insights** | Strategic negotiation | Cultural prompt pitfalls | Both valuable |

---

## 10. CONCLUSION

The Buy-Sell game reveals that **task structure fundamentally changes how language and models interact**:

- **Simple tasks** (Buy-Sell) → High acceptance, language levels playing field, cultural effects amplified
- **Complex tasks** (Negotiation Arena) → Variable acceptance, English advantaged, cultural effects moderate

**Most Important Takeaway:**
> You cannot assume patterns from one negotiation context will transfer to another. The Marwadi catastrophe in Buy-Sell (not seen in Negotiation Arena) and the reversal of English advantage demonstrate that **game structure, language, and model capabilities interact in non-obvious ways**.

**Best Practice:**
Test empirically across multiple languages, models, and task complexities. Cultural and linguistic effects are real, dramatic, and context-dependent.

---

*Analysis based on 321 games across 6 languages and 4 models*
*Data location: NegotiationArena-main/.logs/buysell_final/*