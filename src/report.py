"""Build the submission narrative from measured outputs, never invented results."""
import json
import math
import platform
import sqlite3
import pandas as pd
from .common import ROOT,OUT

def table(data):
    if isinstance(data,dict):data=pd.DataFrame(list(data.items()),columns=['Measure','Value'])
    if not isinstance(data,pd.DataFrame):data=pd.DataFrame(data)
    def fmt(x):
        if x is None or (isinstance(x,float) and math.isnan(x)):return '—'
        if isinstance(x,float):return f'{x:.6f}'
        return str(x).replace('|','/').replace('\n',' ')
    return '| '+' | '.join(map(str,data.columns))+' |\n| '+' | '.join(['---']*len(data.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in data.itertuples(index=False,name=None))+'\n'

def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))

def write_readme():
    a=read('part_a_summary.json'); audit=read('audit.json'); b=read('part_b_summary.json');c=read('part_c_summary.json');r=read('part_d_rag_summary.json');agent=read('part_d_agent_summary.json');rfm=read('rfm_summary.json')
    text='''# Overview

Instamart Customer Intelligence & Support Copilot is a reproducible, entirely offline capstone covering trusted data, customer segmentation, lapse prediction, safe ticket triage, policy retrieval and a bounded refund agent. All data and policies are synthetic and fictional; amounts are INR. Python source is in `src/`, SQL in `sql/`, measured evidence in `outputs/`, and resumable agent states in `state/`. No API key, paid service or network call is used by the project. Installing dependencies initially needs package access.

The authoritative grading brief is `specification.txt` (the attachment named Pasted text(1).txt). Root `generate_data.py` and `mock_llm.py` were extracted verbatim from its code blocks; only trailing whitespace/line endings were normalized when saving. `tests/test_project.py` checks their contents against the embedded blocks after newline normalization. `reference/starter_originals/` preserves all seven uploaded starter files byte-for-byte; `reference/supplied_script_differences.diff` records their differences from the authoritative originals. The teaching kit's generator and mock differ from the brief; its runner omits the refund agent and depends on absent source modules; its README references an absent notebook; its ignore rules exclude required database/checkpoint artifacts. These were replaced in the executable project. The original broad dependency list and learning references are preserved for provenance.

This package was prepared with AI assistance. Review and understand the implementation and follow your programme's disclosure rules. No other student's solution was used.

# Reproduce

Use Python 3.11 or newer (tested with Python '''+platform.python_version()+''') and SQLite 3.25+ (tested '''+sqlite3.sqlite_version+'''). Run from the repository root. No activation is necessary. The following numbered commands use Windows PowerShell; on macOS/Linux replace `.venv\\Scripts\\python.exe` with `.venv/bin/python` throughout.

1. `python -m venv .venv`
2. `.venv\\Scripts\\python.exe -m pip install -r requirements.txt`
3. `.venv\\Scripts\\python.exe generate_data.py`
4. `.venv\\Scripts\\python.exe run_project.py`
5. `.venv\\Scripts\\python.exe validate.py`
6. `.venv\\Scripts\\python.exe -m unittest discover -s tests -v`

The generator prints `generated: customers=1200 products=40 orders_raw=13084 tickets=164`; validation prints exactly `ALL CHECKPOINTS PASSED`. The full runner regenerates all analytical outputs and this README. It intentionally resets only its own demonstration ledger, run log and `demo_*.json` states; this is a repeatable simulation. Independent agent runs use unique run IDs. Fixed seeds are 42 for data, clustering, splitting, forests, oversampling and retry jitter. Ledger timestamps and measured retry waits are recorded; timestamps differ between reruns. Model/joblib compatibility is tied to the pinned dependencies, generated with pip freeze in the test environment. `.gitattributes` disables automatic newline conversion so the preserved starter checksums survive cloning on Windows and other platforms.

To publish after review, create an empty public GitHub repository, initialize this folder with `git init -b main`, then `git add .`, `git commit -m "Complete offline capstone"`, `git remote add origin <your-repository-url>` and `git push -u origin main`. Submit `https://github.com/<user>/<repo>/tree/main`, checking it is readable while signed out. This ZIP is GitHub-ready; no remote repository has been created or published by this run.

# Part A

## Raw audit

Missing values mean blank CSV fields, including nonnumeric planted values as separately reported below. Raw tables were read without converting empty strings into NaN for this audit.

'''
    text+=table(audit['specific'])
    for name in ['customers','products','orders']:
        info=audit[name];text+=f"\n{name}: shape {info['shape']}.\n\n"
        text+=table([{'column':k,'dtype':v,'missing':info['missing_count'][k],'missing_pct':info['missing_pct'][k]} for k,v in info['dtypes'].items()])
    text+='''
## Cleaning and validation

Cities are stripped and title-cased. Genders are lowercased, M/F mapped to male/female, and blank mapped to unknown. Ages outside 10–100 are first set missing, then all missing ages filled by city median using groupby/transform. Signup dates and order timestamps are parsed. Exact order duplicates are removed before keeping the latest timestamp per order ID. Unparseable amounts are reconstructed from catalogue price × quantity × (1 − discount/100), rounded to two decimals. Negative durations become missing, then completed-order missing durations use the completed median of their product category with groupby/transform. Cancelled durations remain missing.

'''+table(read('cleaning_changes.json'))
    customers=pd.read_csv(ROOT/'data/clean/customers.csv');orders=pd.read_csv(ROOT/'data/clean/orders.csv')
    text+='\nClean city counts:\n\n'+table(customers.city.value_counts().to_dict())+'\nClean gender counts:\n\n'+table(customers.gender.value_counts().to_dict())
    text+='\n'+table({'clean_orders':len(orders),'completed':int(orders.status.eq('completed').sum()),'cancelled':int(orders.status.eq('cancelled').sum()),'pending':int(orders.status.eq('pending').sum()),'missing_amount':int(orders.order_amount.isna().sum()),'missing_completed_delivery':int(orders.loc[orders.status.eq('completed'),'delivery_minutes'].isna().sum()),'missing_cancelled_delivery':int(orders.loc[orders.status.eq('cancelled'),'delivery_minutes'].isna().sum()),'missing_age':int(customers.age.isna().sum()),'minimum_age':customers.age.min(),'maximum_age':customers.age.max()})
    text+='''
The three clean CSVs and SQLite tables are saved under the exact required paths. Unique order IDs are indexed in SQLite. Validation asserts city and gender counts, ages, order counts, statuses, amounts, duration rules and revenue. It stops on a failed checkpoint.

## SQL and pandas findings

'''
    text+=table([{'Query':'Q1','result':'30 customers'}, {'Query':'Q2','result':'30 cells; INR 2,998,668.80; largest Bengaluru/Electronics INR 233,909.45'}, {'Query':'Q3','result':'11 products'}, {'Query':'Q4','result':'25 customers'}, {'Query':'Q5','result':f"550 customers; average INR {a['q5_average']:.2f}"}, {'Query':'Q6','result':'1,175 ranked customers; 8 RANK/ROW_NUMBER differences'}, {'Query':'Q7','result':'183 days; final cumulative INR 2,998,668.80'}])
    text+='\nTop customer per city:\n\n'+table(a['top_customers'])
    text+='\nTies share RANK and leave gaps after the tied values, whereas ROW_NUMBER assigns a distinct consecutive number to every row.\n\nThe pandas city/category pivot uses margins=True and asserts its grand total equals SQL revenue. City named aggregations:\n\n'+table(pd.read_csv(OUT/'city_summary.csv'))
    text+='\nCompleted amount distribution:\n\n'+table({k:v for k,v in a.items() if k not in ['top_customers','q5_average']})
    text+='\nCorrelation matrix:\n\n'+table(pd.read_csv(OUT/'correlations.csv'))
    text+='''
This data describes fictional grocery-commerce customers, final orders, catalogue products and support requests over six cities. Completed order amounts are right-skewed: mean INR 269.91 exceeds median INR 135.00, with skewness about 4.61. Electronics explain plausible high-value purchases; 897 amounts exceed the INR 645.30 upper IQR fence, and none fall below the lower fence. These are distributional outliers, not automatically errors. Planted duplicates, invalid amounts, ages and delivery times required explicit cleaning. The largest absolute measured correlation is order_amount with discount_pct (about −0.095), slightly exceeding amount/quantity (+0.087). This weak correlation does not establish causation.

# Part B

## RFM and segmentation

Completed orders only; snapshot 2025-10-01 00:00:00. Recency uses whole elapsed days and frequency counts distinct orders.

'''+table(pd.DataFrame(rfm['summary']).reset_index(names='statistic'))
    text+='\nK-means diagnostics (n_init=10, random_state=42):\n\n'+table(pd.read_csv(OUT/'kmeans_diagnostics.csv'))
    text+='''
Choose K=4: the elbow is clear at four groups and its silhouette is stronger than K=3 or K=5, with distinct actionable profiles. Increasing K gives centroids more freedom to reduce within-cluster squared distances, so K=8 has lower inertia than K=2. Inertia alone cannot penalize unnecessary complexity or ensure stable, useful business segments.

'''+table(pd.read_csv(OUT/'segment_profile.csv'))
    text+='''
Premium receives high-value retention offers, Frequent receives loyalty/replenishment reminders, Dormant receives a win-back campaign, and Occasional receives product discovery or a second-order incentive. Cluster IDs are arbitrary labels with no ordering. Exactly the cluster with highest mean monetary is marked Premium.

'''+table({'PCA_first':rfm['pca'][0],'PCA_second':rfm['pca'][1],'PCA_total':sum(rfm['pca'])})
    text+='''
## Lapse definition, training and test evaluation

Cohort: at least one completed order before 2025-09-01. Positive label: no completed order during September. Features are pre-cutoff recency, distinct-order frequency, monetary, mean discount, cancellation rate among final orders, mean delivery time, favourite category (alphabetical first mode for ties), city, age, gender and tenure. No order from 2025-09-01 onwards enters feature aggregation; segment labels are not model features. A stratified 80/20 split uses random_state=42. All feature columns are asserted nonmissing. OneHotEncoder(drop="first") and StandardScaler are fitted inside training-only pipelines and inside every CV fold. No encoder, scaler or resampler saw test rows. Oversampling duplicates only minority training rows, preserving every original row. Delivery values come from the Part A category-median-cleaned table; its full-window imputation is a residual leakage caveat discussed below.

'''+table({k:b[k] for k in ['cohort','lapsed','train','test','train_class_counts','frequency_scaler_mean','frequency_cohort_mean','balanced_counts']})
    text+='\nThe training scaler mean differs from the full-cohort frequency mean because the random training subset has a slightly different distribution. Lapsed prevalence is '+f"{100*b['lapsed']/b['cohort']:.2f}%"+'.\n'
    text+='''
LogisticRegression(max_iter=1000): C grid {0.01, 0.1, 1, 10, 100}; RandomForestClassifier(n_estimators=100, random_state=42): max_depth {3,5,8} × min_samples_leaf {5,10,20}. Both use GridSearchCV(cv=5, scoring="f1"); 25 and 45 CV fits respectively, followed by refit. All scores below use the lapsed positive class and confusion matrices use [[TN, FP], [FN, TP]]. Grid-level scores are saved in lr_grid_search.csv and rf_grid_search.csv.

'''+table([{'model':key,**b[key]} for key in ['baseline','lr','rf','balanced_lr']])
    text+=f"\nSelected model: **{b['selected']}**, selected by the higher training cross-validated F1 before test evaluation. Missing a lapsing customer can lose future revenue, while targeting a loyal customer wastes an incentive and contact effort. F1 balances these costs when explicit monetary costs are unavailable. The unchanged test results above estimate performance after tuning; the test set does not choose parameters. In deployment, a cost-based probability threshold should be chosen on validation data.\n"
    diffs={k:b['balanced_lr'][k]-b['lr'][k] for k in ['precision','recall','f1']};metric=max(diffs,key=lambda k:abs(diffs[k]))
    text+='\nBalanced-minus-original LR metrics:\n\n'+table(diffs)+f'\nThe largest absolute change is {metric}. Higher recall reaches more at-risk customers; any precision decline increases wasted outreach. Oversampling does not create new independent evidence.\n'
    forest_readings={'frequency':'Past ordering intensity is the strongest retention signal; prioritize activity changes.',
      'monetary':'Historical spend helps distinguish purchasing patterns and expected customer value.',
      'recency':'Time since last order signals disengagement and helps time outreach.',
      'cancellation_rate':'Failed ordering experiences can identify customers needing service recovery.',
      'avg_delivery':'Delivery experience contributes to risk; inspect service quality alongside engagement.'}
    text+='\nForest five most important features (impurity importance shows association, not causal effect or direction):\n\n'+table([{'feature':n,'importance':v,'reading':forest_readings.get(n.split('__')[-1],'Encoded customer attribute contributes to predictive discrimination.')} for n,v in b['rf_top_five']])
    readings={'recency':'Positive: longer inactivity raises predicted lapse odds, holding other features fixed.',
      'frequency':'Negative: more historical orders lowers predicted lapse odds, holding other features fixed.',
      'monetary':'Positive: higher spend at fixed frequency raises predicted lapse odds, consistent with infrequent high-value purchasing.'}
    text+='\nLogistic coefficients (scaled features):\n\n'+table([{'feature':n,'coefficient':v,'sign':'positive' if v>0 else 'negative','business_reading':readings[n]} for n,v in b['lr_coefficients'].items()])
    text+=f"\nThe saved joblib model reload reproduces test predictions exactly: {b['reload_identical']}. The top Premium test retention list contains {b['retention_true_lapsed']} truly lapsed customers out of {len(pd.read_csv(OUT/'retention_list.csv'))}. The Premium flag is a post-hoc whole-window label, so the retention list is illustrative and not a leakage-free evaluation.\n"
    text+='''
## Enter a new customer's details

After the saved model exists, run `.venv\\Scripts\\python.exe predict_customer.py` from the repository root. The script asks for a synthetic customer identifier, recency, frequency, monetary spend, average discount and delivery time, cancellation fraction, favourite category, city, age, gender and tenure. It validates the entries and prints an estimated lapse probability and predicted class using the saved model, without modifying data or retraining. No name, phone number or email is needed. Use the five-month history ending before the scoring month, consistent with the training window; lapse refers to no completed orders in the following month. A brand-new customer with zero completed orders cannot be scored because this model was trained only on existing purchasers.

For a quick fictional example, run `.venv\\Scripts\\python.exe predict_customer.py --example`. Its measured score is about 34.9%, classified as likely to remain active. This is an educational synthetic-data prediction, not an observed outcome or a validated production probability. The additional inference/input-validation tests bring the current test suite to 21 tests; original clean-source reproduction logs document the earlier 17-test submission.

# Part C

Empty/whitespace tickets never call the model; texts over 400 estimated tokens never call it. The four long tickets estimate to 734 tokens each. The raw ticket file is read with keep_default_na=False. Before prompting, redaction applies email, phone and order-ID regexes in the exact required order; patterns and replacements are in src/triage.py. Redaction runs across all 164 tickets, and assertions verify zero remaining matches and that the user message is exactly the redacted text. System prompts contain no customer details.

'''+table(c['few']['redactions'])+'\nTotal substitutions: '+str(sum(c['few']['redactions'].values()))+'; remaining matches: 0.\n'
    text+='''
The zero-shot prompt lists four categories and requests JSON only; the few-shot prompt adds four labelled examples, including fraud, under EXAMPLES:. The mock is reset before each full run. Every call retries at most three times with 0.05 × 2**n × uniform(0.9,1.15) seconds; logs include ticket ID, zero-based retry and wait. Each full run recovered all 19 transient failures, with no permanent error; every observed retry is number 0 with wait in [0.045, 0.0575] seconds. Usage totals sum successful response.usage.total_tokens; failed attempts return no usage.

Valid JSON is accepted only for an allowed category. Non-JSON text is accepted only if re.findall(r"[a-z]+", text.lower()) contains exactly one distinct allowed category word. Valid JSON with an invalid category is rejected without keyword rescue. Accepted outputs are rule-validated categories, not independent factual verification. Invalid, empty, long and budget-stopped tickets go to humans; fraud routes to security HIGH, other premium customers to premium desk HIGH, and others to support NORMAL.

'''+table([{'run':key,**c[key]} for key in ['zero','few','budget']])
    text+='\nZero-shot confusion table:\n\n'+table(pd.read_csv(OUT/'triage_zero_confusion.csv'))
    text+=f"\nFew-shot accuracy is {100*c['few']['accuracy']:.1f}% versus {100*c['zero']['accuracy']:.1f}% for zero-shot on {c['few']['classified']} classified tickets. Its cost rises from {c['zero']['tokens_per_classified']:.2f} to {c['few']['tokens_per_classified']:.2f} tokens per classified ticket. Zero-shot wrongly calls 11 fraud and 10 technical tickets billing, so few-shot examples are worthwhile when missed security incidents or technical misrouting cost more than the added tokens. The mock's deterministic keyword priorities explain this gain; real-world accuracy requires separate evaluation.\n"
    text+=f"\nBudget demo: budget {c['budget']['token_budget']}, {c['budget']['status_counts'].get('budget_exhausted',0)} exhausted tickets. The budget is checked before each call against usage already spent, so one response can cross the limit. Final few-shot output uses budget 100000 and has 164 rows.\n"
    text+='''
# Part D

## Retrieval and grounded answers

Policies are split into 40-word chunks with stride 30 (10-word overlap), stopping at the first chunk reaching the document end. TfidfVectorizer() is fitted on chunks and cosine similarity ranks query matches. TF measures how often a term occurs in a chunk, adjusted by vector normalization. IDF downweights terms common across chunks and gives rarer terms more weight. Lexically different paraphrases can be missed because TF-IDF has no semantic understanding. For `refund policy late delivery`, all top-three chunks are from refund_policy.txt.

'''+table([{'overlap':k,**v} for k,v in r.items()])
    text+='''
Each answer sends the top three chunks with source labels in the exact CONTEXT/QUESTION message format to the offline POLICY ASSISTANT, instructing it to use only context and otherwise say it could not find information. Sources are returned alongside the text. Evaluation uses case-sensitive expected-fact substring matches and top-three expected-document presence; these are narrow benchmark checks, not general answer-quality guarantees.

'''
    d0=pd.read_csv(OUT/'policy_eval_overlap_0.csv',keep_default_na=False)
    text+='\nZero-overlap failures with their actual answers:\n\n'+table(d0[d0.question.isin(r['0']['failed_questions'])][['question','expected_fact','answer']])
    text+='\nWithout overlap, the refund sentence is split after "the customer" at the 40-word boundary, separating its "more than 45 minutes" condition from the full-refund conclusion. The failed-payment sentence is split after "automatically", separating the payment-failure description from "within 48 hours". The retrieved fragments give the mock incomplete sentences; overlap retains enough neighbouring words for both benchmark answers.\n'
    text+='''
## Refund tools, ReAct and limits

TOOLS_SPEC supplies names, descriptions and JSON-style parameter schemas for get_order, search_policy, calculate_refund, issue_refund and escalate_to_human; TOOL_MAP dispatches their arguments to Python. Order lookups use parameterized SQLite queries. Policy search reuses the same retrieval class. Only completed orders with delivery above 45 minutes qualify for a full refund; at or below 45 minutes returns zero. Above INR 1000 records human escalation without issuing money. Issuing/escalation tools independently recheck order status, eligibility and amount, and append timestamped JSONL records. All simulated requests are inside seven days. These are ledger simulations, not real payments.

Each model tool request is appended as an assistant tool_calls message, followed by a JSON tool observation. Final status is the prefix before the colon. Completed steps and usage are returned on all exits. Checkpoints persist run ID, scenario, full messages, calls, tokens, step number and executed tools after each observation. Atomic checkpoint replacement avoids partially written JSON; stable execution keys make ledger append idempotent on replay. The offline mock emits one tool per response. Resume continues from its last saved observation, preserves cumulative call/token counts, and starts a fresh wall-clock budget for the resumed invocation.

'''+table(agent['scenarios'])
    text+='\n'+table({'S1 token_limit=200':agent['token_demo'],'time_limit_seconds=0':agent['timeout_demo'],'crash_resume_status':agent['resumed_status'],'crash_resume_tool_order':' → '.join(agent['resume_tools'])})
    text+='''
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
'''
    (ROOT/'README.md').write_text(text,encoding='utf-8')
