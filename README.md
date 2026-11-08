# Overview

Instamart Customer Intelligence & Support Copilot is a reproducible, entirely offline capstone covering trusted data, customer segmentation, lapse prediction, safe ticket triage, policy retrieval and a bounded refund agent. All data and policies are synthetic and fictional; amounts are INR. Python source is in `src/`, SQL in `sql/`, measured evidence in `outputs/`, and resumable agent states in `state/`. No API key, paid service or network call is used by the project. Installing dependencies initially needs package access.

The authoritative grading brief is `specification.txt` (the attachment named Pasted text(1).txt). Root `generate_data.py` and `mock_llm.py` were extracted verbatim from its code blocks; only trailing whitespace/line endings were normalized when saving. `tests/test_project.py` checks their contents against the embedded blocks after newline normalization. `reference/starter_originals/` preserves all seven uploaded starter files byte-for-byte; `reference/supplied_script_differences.diff` records their differences from the authoritative originals. The teaching kit's generator and mock differ from the brief; its runner omits the refund agent and depends on absent source modules; its README references an absent notebook; its ignore rules exclude required database/checkpoint artifacts. These were replaced in the executable project. The original broad dependency list and learning references are preserved for provenance.

This package was prepared with AI assistance. Review and understand the implementation and follow your programme's disclosure rules. No other student's solution was used.

# Reproduce

Use Python 3.11 or newer (tested with Python 3.12.14) and SQLite 3.25+ (tested 3.53.1). Run from the repository root. No activation is necessary. The following numbered commands use Windows PowerShell; on macOS/Linux replace `.venv\Scripts\python.exe` with `.venv/bin/python` throughout.

1. `python -m venv .venv`
2. `.venv\Scripts\python.exe -m pip install -r requirements.txt`
3. `.venv\Scripts\python.exe generate_data.py`
4. `.venv\Scripts\python.exe run_project.py`
5. `.venv\Scripts\python.exe validate.py`
6. `.venv\Scripts\python.exe -m unittest discover -s tests -v`

The generator prints `generated: customers=1200 products=40 orders_raw=13084 tickets=164`; validation prints exactly `ALL CHECKPOINTS PASSED`. The full runner regenerates all analytical outputs and this README. It intentionally resets only its own demonstration ledger, run log and `demo_*.json` states; this is a repeatable simulation. Independent agent runs use unique run IDs. Fixed seeds are 42 for data, clustering, splitting, forests, oversampling and retry jitter. Ledger timestamps and measured retry waits are recorded; timestamps differ between reruns. Model/joblib compatibility is tied to the pinned dependencies, generated with pip freeze in the test environment.

To publish after review, create an empty public GitHub repository, initialize this folder with `git init -b main`, then `git add .`, `git commit -m "Complete offline capstone"`, `git remote add origin <your-repository-url>` and `git push -u origin main`. Submit `https://github.com/<user>/<repo>/tree/main`, checking it is readable while signed out. This ZIP is GitHub-ready; no remote repository has been created or published by this run.

# Part A

## Raw audit

Missing values mean blank CSV fields, including nonnumeric planted values as separately reported below. Raw tables were read without converting empty strings into NaN for this audit.

| Measure | Value |
| --- | --- |
| raw_rows | 13084 |
| exact_duplicates | 119 |
| pending | 656 |
| distinct_orders | 12309 |
| non_numeric_amounts | 120 |
| negative_delivery | 106 |
| blank_delivery | 2541 |
| blank_age | 61 |
| invalid_age | 6 |
| city_spellings | 24 |
| gender_spellings | 10 |
| blank_gender | 30 |

customers: shape [1200, 5].

| column | dtype | missing | missing_pct |
| --- | --- | --- | --- |
| customer_id | object | 0 | 0.000000 |
| city | object | 0 | 0.000000 |
| signup_date | object | 0 | 0.000000 |
| age | object | 61 | 5.083333 |
| gender | object | 30 | 2.500000 |

products: shape [40, 4].

| column | dtype | missing | missing_pct |
| --- | --- | --- | --- |
| product_id | object | 0 | 0.000000 |
| product_name | object | 0 | 0.000000 |
| category | object | 0 | 0.000000 |
| price_inr | int64 | 0 | 0.000000 |

orders: shape [13084, 9].

| column | dtype | missing | missing_pct |
| --- | --- | --- | --- |
| order_id | object | 0 | 0.000000 |
| customer_id | object | 0 | 0.000000 |
| product_id | object | 0 | 0.000000 |
| quantity | int64 | 0 | 0.000000 |
| order_ts | object | 0 | 0.000000 |
| status | object | 0 | 0.000000 |
| discount_pct | int64 | 0 | 0.000000 |
| order_amount | object | 0 | 0.000000 |
| delivery_minutes | object | 2541 | 19.420666 |

## Cleaning and validation

Cities are stripped and title-cased. Genders are lowercased, M/F mapped to male/female, and blank mapped to unknown. Ages outside 10–100 are first set missing, then all missing ages filled by city median using groupby/transform. Signup dates and order timestamps are parsed. Exact order duplicates are removed before keeping the latest timestamp per order ID. Unparseable amounts are reconstructed from catalogue price × quantity × (1 − discount/100), rounded to two decimals. Negative durations become missing, then completed-order missing durations use the completed median of their product category with groupby/transform. Cancelled durations remain missing.

| Measure | Value |
| --- | --- |
| city_values_changed | 476 |
| gender_values_changed | 950 |
| ages_imputed | 67 |
| exact_rows_removed | 119 |
| earlier_records_removed | 656 |
| amounts_reconstructed | 112 |
| negative_durations_cleared | 105 |
| completed_durations_imputed | 774 |

Clean city counts:

| Measure | Value |
| --- | --- |
| Pune | 227 |
| Delhi | 210 |
| Bengaluru | 197 |
| Chennai | 191 |
| Hyderabad | 191 |
| Mumbai | 184 |

Clean gender counts:

| Measure | Value |
| --- | --- |
| female | 610 |
| male | 560 |
| unknown | 30 |

| Measure | Value |
| --- | --- |
| clean_orders | 12309.000000 |
| completed | 11110.000000 |
| cancelled | 1199.000000 |
| pending | 0.000000 |
| missing_amount | 0.000000 |
| missing_completed_delivery | 0.000000 |
| missing_cancelled_delivery | 1199.000000 |
| missing_age | 0.000000 |
| minimum_age | 18.000000 |
| maximum_age | 60.000000 |

The three clean CSVs and SQLite tables are saved under the exact required paths. Unique order IDs are indexed in SQLite. Validation asserts city and gender counts, ages, order counts, statuses, amounts, duration rules and revenue. It stops on a failed checkpoint.

## SQL and pandas findings

| Query | result |
| --- | --- |
| Q1 | 30 customers |
| Q2 | 30 cells; INR 2,998,668.80; largest Bengaluru/Electronics INR 233,909.45 |
| Q3 | 11 products |
| Q4 | 25 customers |
| Q5 | 550 customers; average INR 2552.06 |
| Q6 | 1,175 ranked customers; 8 RANK/ROW_NUMBER differences |
| Q7 | 183 days; final cumulative INR 2,998,668.80 |

Top customer per city:

| city | customer_id | total_spend | rank | dense_rank | row_number |
| --- | --- | --- | --- | --- | --- |
| Bengaluru | C0082 | 12856.200000 | 1 | 1 | 1 |
| Chennai | C0930 | 10151.400000 | 1 | 1 | 1 |
| Delhi | C0291 | 13742.600000 | 1 | 1 | 1 |
| Hyderabad | C0743 | 12730.050000 | 1 | 1 | 1 |
| Mumbai | C0693 | 14766.550000 | 1 | 1 | 1 |
| Pune | C0426 | 10377.100000 | 1 | 1 | 1 |

Ties share RANK and leave gaps after the tied values, whereas ROW_NUMBER assigns a distinct consecutive number to every row.

The pandas city/category pivot uses margins=True and asserts its grand total equals SQL revenue. City named aggregations:

| city | sum | mean | count |
| --- | --- | --- | --- |
| Bengaluru | 523643.400000 | 276.474868 | 1894 |
| Chennai | 463629.150000 | 272.723029 | 1700 |
| Delhi | 502829.800000 | 243.619089 | 2064 |
| Hyderabad | 528589.900000 | 284.954124 | 1855 |
| Mumbai | 479058.500000 | 296.998450 | 1613 |
| Pune | 500918.050000 | 252.478856 | 1984 |

Completed amount distribution:

| Measure | Value |
| --- | --- |
| mean | 269.907183 |
| median | 135.000000 |
| skewness | 4.608827 |
| q1 | 64.800000 |
| q3 | 297.000000 |
| lower_fence | -283.500000 |
| upper_fence | 645.300000 |
| below_fence | 0.000000 |
| above_fence | 897.000000 |

Correlation matrix:

| Unnamed: 0 | order_amount | discount_pct | quantity | delivery_minutes |
| --- | --- | --- | --- | --- |
| order_amount | 1.000000 | -0.095106 | 0.087294 | -0.003135 |
| discount_pct | -0.095106 | 1.000000 | -0.000359 | -0.009808 |
| quantity | 0.087294 | -0.000359 | 1.000000 | -0.012623 |
| delivery_minutes | -0.003135 | -0.009808 | -0.012623 | 1.000000 |

This data describes fictional grocery-commerce customers, final orders, catalogue products and support requests over six cities. Completed order amounts are right-skewed: mean INR 269.91 exceeds median INR 135.00, with skewness about 4.61. Electronics explain plausible high-value purchases; 897 amounts exceed the INR 645.30 upper IQR fence, and none fall below the lower fence. These are distributional outliers, not automatically errors. Planted duplicates, invalid amounts, ages and delivery times required explicit cleaning. The largest absolute measured correlation is order_amount with discount_pct (about −0.095), slightly exceeding amount/quantity (+0.087). This weak correlation does not establish causation.

# Part B

## RFM and segmentation

Completed orders only; snapshot 2025-10-01 00:00:00. Recency uses whole elapsed days and frequency counts distinct orders.

| statistic | recency | frequency | monetary |
| --- | --- | --- | --- |
| count | 1175.000000 | 1175.000000 | 1175.000000 |
| mean | 33.558298 | 9.455319 | 2552.058553 |
| std | 38.433705 | 8.009971 | 2367.305104 |
| min | 0.000000 | 1.000000 | 12.000000 |
| 25% | 6.000000 | 3.000000 | 505.450000 |
| 50% | 19.000000 | 5.000000 | 2286.500000 |
| 75% | 46.000000 | 17.000000 | 3751.525000 |
| max | 180.000000 | 33.000000 | 14766.550000 |

K-means diagnostics (n_init=10, random_state=42):

| k | inertia | silhouette |
| --- | --- | --- |
| 2 | 1820.760553 | 0.464972 |
| 3 | 1268.234587 | 0.445161 |
| 4 | 743.071108 | 0.525544 |
| 5 | 617.366636 | 0.480495 |
| 6 | 503.170622 | 0.481872 |
| 7 | 408.460454 | 0.416337 |
| 8 | 366.503487 | 0.412329 |

Choose K=4: the elbow is clear at four groups and its silhouette is stronger than K=3 or K=5, with distinct actionable profiles. Increasing K gives centroids more freedom to reduce within-cluster squared distances, so K=8 has lower inertia than K=2. Inertia alone cannot penalize unnecessary complexity or ensure stable, useful business segments.

| segment_id | size | recency | frequency | monetary | segment_name |
| --- | --- | --- | --- | --- | --- |
| 0 | 436 | 28.862385 | 3.772936 | 1057.618234 | Occasional |
| 1 | 138 | 24.115942 | 6.768116 | 7129.695652 | Premium |
| 2 | 419 | 8.766110 | 19.403341 | 3410.020525 | Frequent |
| 3 | 182 | 109.043956 | 2.203297 | 685.992582 | Dormant |

Premium receives high-value retention offers, Frequent receives loyalty/replenishment reminders, Dormant receives a win-back campaign, and Occasional receives product discovery or a second-order incentive. Cluster IDs are arbitrary labels with no ordering. Exactly the cluster with highest mean monetary is marked Premium.

| Measure | Value |
| --- | --- |
| PCA_first | 0.634279 |
| PCA_second | 0.218038 |
| PCA_total | 0.852316 |

## Lapse definition, training and test evaluation

Cohort: at least one completed order before 2025-09-01. Positive label: no completed order during September. Features are pre-cutoff recency, distinct-order frequency, monetary, mean discount, cancellation rate among final orders, mean delivery time, favourite category (alphabetical first mode for ties), city, age, gender and tenure. No order from 2025-09-01 onwards enters feature aggregation; segment labels are not model features. A stratified 80/20 split uses random_state=42. All feature columns are asserted nonmissing. OneHotEncoder(drop="first") and StandardScaler are fitted inside training-only pipelines and inside every CV fold. No encoder, scaler or resampler saw test rows. Oversampling duplicates only minority training rows, preserving every original row. Delivery values come from the Part A category-median-cleaned table; its full-window imputation is a residual leakage caveat discussed below.

| Measure | Value |
| --- | --- |
| cohort | 1156 |
| lapsed | 443 |
| train | 924 |
| test | 232 |
| train_class_counts | {'0': 570, '1': 354} |
| frequency_scaler_mean | 8.114719 |
| frequency_cohort_mean | 8.028547 |
| balanced_counts | {'1': 570, '0': 570} |

The training scaler mean differs from the full-cohort frequency mean because the random training subset has a slightly different distribution. Lapsed prevalence is 38.32%.

LogisticRegression(max_iter=1000): C grid {0.01, 0.1, 1, 10, 100}; RandomForestClassifier(n_estimators=100, random_state=42): max_depth {3,5,8} × min_samples_leaf {5,10,20}. Both use GridSearchCV(cv=5, scoring="f1"); 25 and 45 CV fits respectively, followed by refit. All scores below use the lapsed positive class and confusion matrices use [[TN, FP], [FN, TP]]. Grid-level scores are saved in lr_grid_search.csv and rf_grid_search.csv.

| model | precision | recall | f1 | accuracy | confusion_matrix | best_params | cv_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.000000 | 0.000000 | 0.000000 | 0.616379 | [[143, 0], [89, 0]] | — | — |
| lr | 0.644444 | 0.651685 | 0.648045 | 0.728448 | [[111, 32], [31, 58]] | {'model__C': 100} | 0.650553 |
| rf | 0.629213 | 0.629213 | 0.629213 | 0.715517 | [[110, 33], [33, 56]] | {'model__max_depth': 5, 'model__min_samples_leaf': 5} | 0.682877 |
| balanced_lr | 0.589147 | 0.853933 | 0.697248 | 0.715517 | [[90, 53], [13, 76]] | — | — |

Selected model: **rf**, selected by the higher training cross-validated F1 before test evaluation. Missing a lapsing customer can lose future revenue, while targeting a loyal customer wastes an incentive and contact effort. F1 balances these costs when explicit monetary costs are unavailable. The unchanged test results above estimate performance after tuning; the test set does not choose parameters. In deployment, a cost-based probability threshold should be chosen on validation data.

Balanced-minus-original LR metrics:

| Measure | Value |
| --- | --- |
| precision | -0.055297 |
| recall | 0.202247 |
| f1 | 0.049203 |

The largest absolute change is recall. Higher recall reaches more at-risk customers; any precision decline increases wasted outreach. Oversampling does not create new independent evidence.

Forest five most important features (impurity importance shows association, not causal effect or direction):

| feature | importance | reading |
| --- | --- | --- |
| num__frequency | 0.317188 | Past ordering intensity is the strongest retention signal; prioritize activity changes. |
| num__monetary | 0.260046 | Historical spend helps distinguish purchasing patterns and expected customer value. |
| num__recency | 0.112350 | Time since last order signals disengagement and helps time outreach. |
| num__cancellation_rate | 0.072730 | Failed ordering experiences can identify customers needing service recovery. |
| num__avg_delivery | 0.049232 | Delivery experience contributes to risk; inspect service quality alongside engagement. |

Logistic coefficients (scaled features):

| feature | coefficient | sign | business_reading |
| --- | --- | --- | --- |
| recency | 0.039585 | positive | Positive: longer inactivity raises predicted lapse odds, holding other features fixed. |
| frequency | -1.698239 | negative | Negative: more historical orders lowers predicted lapse odds, holding other features fixed. |
| monetary | 0.284573 | positive | Positive: higher spend at fixed frequency raises predicted lapse odds, consistent with infrequent high-value purchasing. |

The saved joblib model reload reproduces test predictions exactly: True. The top Premium test retention list contains 0 truly lapsed customers out of 10. The Premium flag is a post-hoc whole-window label, so the retention list is illustrative and not a leakage-free evaluation.

# Part C

Empty/whitespace tickets never call the model; texts over 400 estimated tokens never call it. The four long tickets estimate to 734 tokens each. The raw ticket file is read with keep_default_na=False. Before prompting, redaction applies email, phone and order-ID regexes in the exact required order; patterns and replacements are in src/triage.py. Redaction runs across all 164 tickets, and assertions verify zero remaining matches and that the user message is exactly the redacted text. System prompts contain no customer details.

| Measure | Value |
| --- | --- |
| email | 55 |
| phone | 91 |
| order | 132 |

Total substitutions: 278; remaining matches: 0.

The zero-shot prompt lists four categories and requests JSON only; the few-shot prompt adds four labelled examples, including fraud, under EXAMPLES:. The mock is reset before each full run. Every call retries at most three times with 0.05 × 2**n × uniform(0.9,1.15) seconds; logs include ticket ID, zero-based retry and wait. Each full run recovered all 19 transient failures, with no permanent error; every observed retry is number 0 with wait in [0.045, 0.0575] seconds. Usage totals sum successful response.usage.total_tokens; failed attempts return no usage.

Valid JSON is accepted only for an allowed category. Non-JSON text is accepted only if re.findall(r"[a-z]+", text.lower()) contains exactly one distinct allowed category word. Valid JSON with an invalid category is rejected without keyword rescue. Accepted outputs are rule-validated categories, not independent factual verification. Invalid, empty, long and budget-stopped tickets go to humans; fraud routes to security HIGH, other premium customers to premium desk HIGH, and others to support NORMAL.

| run | status_counts | classified | correct | accuracy | total_tokens | tokens_per_classified | retries | redactions | remaining_pii | token_budget |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| zero | {'classified_json': 135, 'classified_keyword': 12, 'flagged_invalid_output': 9, 'flagged_long': 4, 'skipped_empty': 4} | 147 | 126 | 0.857143 | 8932 | 60.761905 | 19 | {'email': 55, 'phone': 91, 'order': 132} | 0 | — |
| few | {'classified_json': 135, 'classified_keyword': 12, 'flagged_invalid_output': 9, 'flagged_long': 4, 'skipped_empty': 4} | 147 | 147 | 1.000000 | 14080 | 95.782313 | 19 | {'email': 55, 'phone': 91, 'order': 132} | 0 | — |
| budget | {'classified_json': 91, 'budget_exhausted': 51, 'classified_keyword': 8, 'flagged_invalid_output': 6, 'skipped_empty': 4, 'flagged_long': 4} | 99 | 84 | 0.848485 | 6023 | 60.838384 | 13 | {'email': 55, 'phone': 91, 'order': 132} | 0 | 6000.000000 |

Zero-shot confusion table:

| true | billing | delivery | fraud | technical |
| --- | --- | --- | --- | --- |
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 11 | 0 | 23 | 0 |
| technical | 10 | 0 | 0 | 29 |

Few-shot accuracy is 100.0% versus 85.7% for zero-shot on 147 classified tickets. Its cost rises from 60.76 to 95.78 tokens per classified ticket. Zero-shot wrongly calls 11 fraud and 10 technical tickets billing, so few-shot examples are worthwhile when missed security incidents or technical misrouting cost more than the added tokens. The mock's deterministic keyword priorities explain this gain; real-world accuracy requires separate evaluation.

Budget demo: budget 6000, 51 exhausted tickets. The budget is checked before each call against usage already spent, so one response can cross the limit. Final few-shot output uses budget 100000 and has 164 rows.

# Part D

## Retrieval and grounded answers

Policies are split into 40-word chunks with stride 30 (10-word overlap), stopping at the first chunk reaching the document end. TfidfVectorizer() is fitted on chunks and cosine similarity ranks query matches. TF measures how often a term occurs in a chunk, adjusted by vector normalization. IDF downweights terms common across chunks and gives rarer terms more weight. Lexically different paraphrases can be missed because TF-IDF has no semantic understanding. For `refund policy late delivery`, all top-three chunks are from refund_policy.txt.

| overlap | chunks | by_source | retrieval_hits | fact_hits | fallbacks | failed_questions |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 15 | {'delivery_policy.txt': 3, 'escalation_policy.txt': 3, 'payment_policy.txt': 3, 'privacy_policy.txt': 2, 'refund_policy.txt': 4} | 12 | 12 | 2 | [] |
| 0 | 13 | {'delivery_policy.txt': 3, 'escalation_policy.txt': 2, 'payment_policy.txt': 3, 'privacy_policy.txt': 2, 'refund_policy.txt': 3} | 12 | 10 | 2 | ['What delivery time makes a completed order eligible for a full refund?', 'How soon is money returned when a payment fails but is deducted?'] |

Each answer sends the top three chunks with source labels in the exact CONTEXT/QUESTION message format to the offline POLICY ASSISTANT, instructing it to use only context and otherwise say it could not find information. Sources are returned alongside the text. Evaluation uses case-sensitive expected-fact substring matches and top-three expected-document presence; these are narrow benchmark checks, not general answer-quality guarantees.


Zero-overlap failures with their actual answers:

| question | expected_fact | answer |
| --- | --- | --- |
| What delivery time makes a completed order eligible for a full refund? | more than 45 minutes | receives a full refund of the order amount. |
| How soon is money returned when a payment fails but is deducted? | 48 hours | If a payment fails but money is deducted, the amount is returned automatically   1000 must be approved by a human support lead before money is returned. |

Without overlap, the refund sentence is split after "the customer" at the 40-word boundary, separating its "more than 45 minutes" condition from the full-refund conclusion. The failed-payment sentence is split after "automatically", separating the payment-failure description from "within 48 hours". The retrieved fragments give the mock incomplete sentences; overlap retains enough neighbouring words for both benchmark answers.

## Refund tools, ReAct and limits

TOOLS_SPEC supplies names, descriptions and JSON-style parameter schemas for get_order, search_policy, calculate_refund, issue_refund and escalate_to_human; TOOL_MAP dispatches their arguments to Python. Order lookups use parameterized SQLite queries. Policy search reuses the same retrieval class. Only completed orders with delivery above 45 minutes qualify for a full refund; at or below 45 minutes returns zero. Above INR 1000 records human escalation without issuing money. Issuing/escalation tools independently recheck order status, eligibility and amount, and append timestamped JSONL records. All simulated requests are inside seven days. These are ledger simulations, not real payments.

Each model tool request is appended as an assistant tool_calls message, followed by a JSON tool observation. Final status is the prefix before the colon. Completed steps and usage are returned on all exits. Checkpoints persist run ID, scenario, full messages, calls, tokens, step number and executed tools after each observation. Atomic checkpoint replacement avoids partially written JSON; stable execution keys make ledger append idempotent on replay. The offline mock emits one tool per response. Resume continues from its last saved observation, preserves cumulative call/token counts, and starts a fresh wall-clock budget for the resumed invocation.

| scenario_id | status | llm_calls | tokens_used | refund_inr |
| --- | --- | --- | --- | --- |
| S1 | REFUND | 5 | 952 | 128.250000 |
| S2 | REFUND | 5 | 952 | 135.000000 |
| S3 | ESCALATE | 5 | 939 | 1999.000000 |
| S4 | DENIED | 4 | 649 | 0.000000 |
| S5 | DENIED | 3 | 370 | 0.000000 |
| S6 | NOT_FOUND | 2 | 94 | 0.000000 |
| S7 | FALLBACK_MAX_ITERATIONS | 6 | 3066 | 0.000000 |

| Measure | Value |
| --- | --- |
| S1 token_limit=200 | FALLBACK_TOKEN_LIMIT |
| time_limit_seconds=0 | FALLBACK_TIMEOUT |
| crash_resume_status | REFUND |
| crash_resume_tool_order | get_order → search_policy → calculate_refund → issue_refund |

S7 reaches all six calls under the default 6000-token limit, then FALLBACK_MAX_ITERATIONS. Token usage is checked immediately after each model response; an over-limit requested tool is never executed. Timeout uses monotonic elapsed >= limit at the top of the iteration; zero seconds stops before any call. Every fallback includes a human-handoff summary and steps so far. The S1 simulated crash occurs immediately after tool two and its saved checkpoint; resume yields REFUND INR 128.25 with exactly four tool executions total and none repeated. `agent_runs.jsonl` has seven scenarios plus token, timeout and resumed demonstrations (ten completed runs); `refund_ledger.jsonl` includes issued S1/S2, escalated S3 and the independent resumed S1 demonstration. Failed/crashed partial work is represented by state, not a fake completed run.

## ReAct versus plan-and-execute

S1 uses five model calls, S5 three and S6 two (tokens are measured in the table above). ReAct pays repeated prompt/history tokens but can inspect each tool observation before choosing the next action. NOT_FOUND stops after lookup; a cancelled order stops after policy retrieval without a refund calculation. Plan-and-execute could request the full tool sequence in one model call, run deterministic dependencies locally and replan only when an observation invalidates that plan, reducing repeated model tokens. Unexpected orders would need explicit branch rules or an additional planning call. For this small fixed refund workflow I would choose deterministic execution of a validated plan with error branches and the same independent tool guards, checkpoints and limits: it is cheaper and easier to audit. ReAct remains useful when the next investigation genuinely depends on open-ended observations.

# Limitations and Next Steps

- The Premium flag is a post-hoc whole-window label, so the retention list is illustrative and not a leakage-free evaluation.
- Features aggregate only pre-cutoff orders, but Part A imputes delivery medians across the full window as required by the brief. This lets September influence missing delivery values indirectly. A production temporal evaluation should fit imputation on pre-cutoff training history and apply it forward; this package follows the prescribed Part A/Part B interface and discloses the caveat.
- Synthetic labels, deterministic mocks and substring grounding tests do not establish production accuracy, fairness or policy coverage. Add held-out real-world evaluation, drift monitoring and review before use. Category feature importances and logistic signs are associations.
- Triage privacy regexes cover only specified email/phone/order forms. Real privacy protection needs broader PII detection, access control and retention limits.
- Checkpoint resume is demonstrated for one-tool responses and crash-after-observation. Production distributed execution needs transactional tool/outbox state and cross-run business idempotency, not just per-run execution keys; this demo can intentionally refund the same synthetic order in different runs.
- The simulated ledger does not move money. No real customer, support team or payment service was contacted.
- The grading brief's two supplied scripts are authoritative; uploaded teaching-kit variants are preserved separately and deliberately not executed as the grading source. The original /mnt/data path is unavailable on this Windows host; the same named attachment was retrieved from the referenced chat.
- Best hyperparameters differ from the illustrative reference: LR C=100 and forest depth=5/leaf=5 in this implementation, with test F1 about 0.648 and 0.629. These were measured rather than forced to match the example; the specified grids, seeds and minimum F1 are respected. All exact data, triage and refund checkpoints and estimator tolerances pass.
- Host setup deviation: Windows blocked the latest pandas binary, so the tested environment uses the pinned stable versions in requirements.txt. The bundled Python lacked ensurepip, so its isolated environment was bootstrapped using the bundled pip with --python. Ordinary Python installations support the documented venv command. Clean-source reproduction reuses that isolated pinned environment; it does not claim a second independent environment installation. Captured pipeline, unittest and clean-source reproduction logs are in outputs/.
- A downloadable repository package is complete locally; public GitHub publication and checking public accessibility remain user actions. No unverified public repository link is claimed. Exact timestamps, serialized model bytes and floating-point rounding can vary by platform; pinned dependencies and numerical assertions capture substantive reproducibility.
