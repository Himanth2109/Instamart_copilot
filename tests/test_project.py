"""Acceptance and adversarial boundary checks; no network or paid service."""
import contextlib
import hashlib
import io
import json
import sqlite3
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import joblib
from src.common import ROOT,OUT
from src.data_pipeline import validate
from src.customer_models import build_features
from src.triage import redact,validate_output,run_pipeline,ZERO,FEW,call_with_retry,PATTERNS
from src.policy_rag import PolicyRAG
from src import refund_agent as agent
import mock_llm

def summary(name):return json.loads((OUT/name).read_text(encoding='utf-8'))

class SuppliedFiles(unittest.TestCase):
    def test_authoritative_code_is_unchanged(self):
        spec=(ROOT/'specification.txt').read_text(encoding='utf-8')
        for name,end in [('generate_data.py','\nmock_llm.py'),('mock_llm.py','Required Repository Contents')]:
            start=spec.index('"""'+name);finish=spec.index(end,start)
            self.assertEqual((ROOT/name).read_text(encoding='utf-8').rstrip(),spec[start:finish].rstrip())

    def test_starter_preservation(self):
        m=json.loads((ROOT/'reference/starter_sha256.json').read_text())
        self.assertEqual(len(m),7)
        for name,digest in m.items():self.assertEqual(hashlib.sha256((ROOT/'reference/starter_originals'/name).read_bytes()).hexdigest(),digest)

class Foundation(unittest.TestCase):
    def test_exact_checkpoints(self):
        s=io.StringIO()
        with contextlib.redirect_stdout(s):validate()
        self.assertEqual(s.getvalue(),'ALL CHECKPOINTS PASSED\n')

    def test_raw_audit(self):
        self.assertEqual(summary('audit.json')['specific'],{'raw_rows':13084,'exact_duplicates':119,'pending':656,'distinct_orders':12309,
          'non_numeric_amounts':120,'negative_delivery':106,'blank_delivery':2541,'blank_age':61,'invalid_age':6,'city_spellings':24,'gender_spellings':10,'blank_gender':30})

    def test_all_sql_results(self):
        queries=[q for q in (ROOT/'sql/queries.sql').read_text().split(';') if q.strip()]
        with sqlite3.connect(ROOT/'data/instamart.db') as con:results=[pd.read_sql_query(q,con) for q in queries]
        self.assertEqual([len(d) for d in results],[30,30,11,25,550,1175,183])
        self.assertAlmostEqual(results[1].revenue.sum(),2998668.8,places=2)
        self.assertEqual(results[1].iloc[0]['city'],'Bengaluru');self.assertEqual(results[1].iloc[0]['category'],'Electronics')
        self.assertAlmostEqual(results[1].iloc[0].revenue,233909.45,places=2)
        self.assertEqual(int((results[5]['rank']!=results[5].row_number).sum()),8)
        self.assertAlmostEqual(results[6].iloc[-1].cumulative_revenue,2998668.8,places=2)

class Models(unittest.TestCase):
    def test_temporal_features_and_model_reload(self):
        x,y=build_features();self.assertEqual((len(x),int(y.sum())),(1156,443));self.assertFalse(x.isna().any().any())
        self.assertFalse(any('segment' in k for k in x))
        from sklearn.model_selection import train_test_split
        _,xv,_,yv=train_test_split(x,y,test_size=.2,stratify=y,random_state=42)
        model=joblib.load(OUT/'lapse_model.joblib');saved=pd.read_csv(OUT/'test_predictions.csv')
        np.testing.assert_array_equal(model.predict(xv),saved.prediction)
        self.assertEqual(list(xv.index),saved.customer_id.tolist())
        self.assertGreaterEqual(summary('part_b_summary.json')[summary('part_b_summary.json')['selected']]['f1'],.55)

    def test_premium_and_retention(self):
        s=pd.read_csv(OUT/'customer_segments.csv');p=s.groupby('segment_id').monetary.mean().idxmax()
        np.testing.assert_array_equal(s.is_premium,s.segment_id.eq(p))
        ret=pd.read_csv(OUT/'retention_list.csv');self.assertEqual(len(ret),10)
        self.assertTrue(set(ret.customer_id)<=set(s.loc[s.is_premium,'customer_id']))
        self.assertTrue(ret.lapse_probability.is_monotonic_decreasing)

    def test_clustering_tolerances(self):
        k=pd.read_csv(OUT/'kmeans_diagnostics.csv')
        np.testing.assert_allclose(k.inertia,[1820.8,1268.2,743.1,617.4,503.2,408.5,366.5],rtol=.05)
        np.testing.assert_allclose(k.silhouette,[.465,.445,.526,.481,.482,.416,.412],atol=.05)
        self.assertEqual(summary('part_b_summary.json')['balanced_counts'],{'0':570,'1':570})

class Triage(unittest.TestCase):
    def test_privacy_and_strict_output_validation(self):
        text,counts=redact('Email a.b+tag@example.com phone 9876543210 ORD-123456')
        self.assertEqual(counts,{'email':1,'phone':1,'order':1})
        self.assertEqual(text,'Email [EMAIL] phone [PHONE] [ORDER_ID]')
        self.assertEqual(redact(text)[1],{'email':0,'phone':0,'order':0})
        for text,expected in [('The customer has a billing-related problem.',('classified_keyword','billing')),
          ('billing and fraud',('flagged_invalid_output','')),('{"category":"shipping","reason":"billing"}',('flagged_invalid_output','')),
          ('[]',('flagged_invalid_output','')),('{"category":[]}',('flagged_invalid_output','')),
          ('{"category":"fraud"}',('classified_json','fraud'))]:self.assertEqual(validate_output(text),expected)

    def test_retry_limits_and_exponential_jitter(self):
        logs=[]
        import random
        with patch('src.triage.mock_llm.chat',side_effect=mock_llm.TransientLLMError('test')) as call,patch('src.triage.time.sleep'):
            with self.assertRaises(mock_llm.TransientLLMError):call_with_retry([], 'test',logs,random.Random(42))
        self.assertEqual(call.call_count,4);self.assertEqual([v['retry_number'] for v in logs],[0,1,2])
        for n,e in enumerate(logs):self.assertTrue(.05*2**n*.9<=e['wait_seconds']<=.05*2**n*1.15)

    def test_no_model_call_when_budget_zero(self):
        with patch('src.triage.mock_llm.chat',side_effect=AssertionError('unexpected call')):
            d,s=run_pipeline(ZERO,0)
        self.assertEqual(d.status.value_counts().to_dict(),{'budget_exhausted':156,'flagged_long':4,'skipped_empty':4})
        self.assertTrue(d.route.eq('human_review').all())

    def test_measured_statuses_privacy_retries_and_accuracy(self):
        s=summary('part_c_summary.json')
        for key in ['zero','few']:
            self.assertEqual(s[key]['status_counts'],{'classified_json':135,'classified_keyword':12,'flagged_invalid_output':9,'flagged_long':4,'skipped_empty':4})
            self.assertEqual(s[key]['retries'],19);self.assertEqual(sum(s[key]['redactions'].values()),278)
        self.assertAlmostEqual(s['zero']['accuracy'],126/147);self.assertEqual(s['few']['accuracy'],1)
        self.assertGreater(s['budget']['status_counts']['budget_exhausted'],0)
        for name in ['triage_zero_retry_log.json','triage_results_retry_log.json']:
            for entry in summary(name):self.assertTrue(.045<=entry['wait_seconds']<=.0575)

    def test_final_routing(self):
        d=pd.read_csv(OUT/'triage_results.csv',keep_default_na=False);s=pd.read_csv(OUT/'customer_segments.csv').set_index('customer_id')
        self.assertEqual(len(d),164)
        for row in d.itertuples():
            if not row.status.startswith('classified_'):expected=('human_review','NORMAL')
            elif row.category=='fraud':expected=('security_team','HIGH')
            elif row.customer_id in s.index and s.loc[row.customer_id,'is_premium']:expected=('premium_desk','HIGH')
            else:expected=('support_queue','NORMAL')
            self.assertEqual((row.route,row.priority),expected)

class RetrievalAndAgent(unittest.TestCase):
    def test_chunking_and_grounding(self):
        r=summary('part_d_rag_summary.json');self.assertEqual(r['10']['chunks'],15)
        self.assertEqual([r['10'][k] for k in ['retrieval_hits','fact_hits','fallbacks']],[12,12,2])
        self.assertLess(r['0']['fact_hits'],12)
        self.assertEqual([x['source'] for x in PolicyRAG().retrieve('refund policy late delivery')],['refund_policy.txt']*3)

    def test_tool_boundaries(self):
        self.assertEqual(agent.get_order('ORD-999999'),{'error':'order not found'})
        self.assertEqual(agent.get_order("' OR 1=1 --"),{'error':'order not found'})
        self.assertEqual(agent.calculate_refund(100,45),{'refund_inr':0})
        self.assertEqual(agent.calculate_refund(100,46),{'refund_inr':100})
        self.assertEqual(agent.calculate_refund(100,None),{'refund_inr':0})
        with self.assertRaises(ValueError):agent.calculate_refund(float('nan'),50)
        rows=pd.read_csv(ROOT/'data/raw/agent_scenarios.csv')
        import re
        big=re.search(r'ORD-\d{6}',rows.iloc[2].message).group()
        with self.assertRaises(ValueError):agent.issue_refund(big,1999)
        cancel=re.search(r'ORD-\d{6}',rows.iloc[4].message).group()
        with self.assertRaises(ValueError):agent.issue_refund(cancel,1)

    def test_scenarios_limits_and_crash_resume(self):
        r=summary('part_d_agent_summary.json');s=r['scenarios']
        self.assertEqual([x['status'] for x in s],['REFUND','REFUND','ESCALATE','DENIED','DENIED','NOT_FOUND','FALLBACK_MAX_ITERATIONS'])
        self.assertEqual([x['refund_inr'] for x in s],[128.25,135,1999,0,0,0,0])
        self.assertEqual([x['llm_calls'] for x in s],[5,5,5,4,3,2,6])
        self.assertEqual(r['token_demo'],'FALLBACK_TOKEN_LIMIT');self.assertEqual(r['timeout_demo'],'FALLBACK_TIMEOUT')
        state=json.loads((ROOT/'state/demo_crash.json').read_text())
        self.assertEqual(state['tools_executed'],['get_order','search_policy','calculate_refund','issue_refund'])
        self.assertEqual(state['status'],'REFUND');self.assertEqual(state['steps'][-1]['observation']['refund_inr'],128.25)
        # Resuming a finished run must neither call the model nor append a ledger entry.
        before=(OUT/'refund_ledger.jsonl').read_bytes()
        with patch('src.refund_agent.mock_llm.chat',side_effect=AssertionError('unexpected call')):
            again=agent.resume_agent('demo_crash')
        self.assertEqual(again['status'],'REFUND');self.assertEqual(before,(OUT/'refund_ledger.jsonl').read_bytes())
        self.assertEqual(len((OUT/'agent_runs.jsonl').read_text().splitlines()),10)

    def test_limits_stop_before_tool_and_before_first_call(self):
        token=json.loads((ROOT/'state/demo_token.json').read_text());timeout=json.loads((ROOT/'state/demo_timeout.json').read_text())
        self.assertGreater(token['tokens_used'],200)
        self.assertLess(len(token['steps']),token['llm_calls'])
        self.assertEqual(timeout['steps'],[]);self.assertEqual(timeout['llm_calls'],0)

if __name__=='__main__':unittest.main()
