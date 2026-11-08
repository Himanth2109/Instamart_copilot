"""Privacy-first offline classification, retries, strict validation and routing."""
import json
import random
import re
import time
import pandas as pd
import mock_llm
from .common import ROOT,OUT,save_json

MAX_INPUT_TOKENS=400
BASE_DELAY=.05
PATTERNS=[('email',r'[\w.+-]+@[\w-]+\.[\w.]+','[EMAIL]'),('phone',r'\b[6-9]\d{9}\b','[PHONE]'),('order',r'ORD-\d{6}','[ORDER_ID]')]
CATEGORIES={'billing','delivery','technical','fraud'}
ZERO='CLASSIFY support tickets into billing, delivery, technical, fraud. Reply JSON only: {"category": "...", "reason": "..."}. Choose the category matching the customer problem.'
FEW=ZERO+'\nEXAMPLES:\nUnauthorized payment on my card: {"category":"fraud","reason":"unauthorized transaction"}\nThe app freezes on payment: {"category":"technical","reason":"app error"}\nDelivery was late: {"category":"delivery","reason":"late delivery"}\nWrong invoice amount: {"category":"billing","reason":"incorrect bill"}'

def redact(text):
    counts={}
    for kind,pattern,placeholder in PATTERNS:text,counts[kind]=re.subn(pattern,placeholder,text)
    assert not any(re.search(p,text) for _,p,_ in PATTERNS)
    return text,counts

def validate_output(text):
    try:
        data=json.loads(text)
    except (json.JSONDecodeError,TypeError):
        found=set(re.findall(r'[a-z]+',(text or '').lower())) & CATEGORIES
        return ('classified_keyword',found.pop()) if len(found)==1 else ('flagged_invalid_output','')
    category=data.get('category') if isinstance(data,dict) else None
    return ('classified_json',category) if isinstance(category,str) and category in CATEGORIES else ('flagged_invalid_output','')

def call_with_retry(messages,ticket_id,retries,rng):
    for attempt in range(4):
        try:return mock_llm.chat(messages)
        except mock_llm.TransientLLMError:
            if attempt==3: raise
            wait=BASE_DELAY*2**attempt*rng.uniform(.9,1.15)
            entry={'ticket_id':ticket_id,'retry_number':attempt,'wait_seconds':wait};retries.append(entry)
            print(f'{ticket_id} retry {attempt}: wait {wait:.6f}s')
            time.sleep(wait)

def run_pipeline(prompt=FEW,token_budget=None,output_name=None):
    tickets=pd.read_csv(ROOT/'data/raw/tickets.csv',keep_default_na=False)
    segments=pd.read_csv(OUT/'customer_segments.csv').set_index('customer_id')
    rows=[];total=0;retries=[];rng=random.Random(42);redactions={k:0 for k,_,_ in PATTERNS};truth=[];pred=[]
    mock_llm.reset()
    for t in tickets.itertuples(index=False):
        text,counts=redact(t.text)
        for k,v in counts.items():redactions[k]+=v
        category='';used=0
        if not t.text.strip():status='skipped_empty'
        elif mock_llm.est_tokens(t.text)>MAX_INPUT_TOKENS:status='flagged_long'
        elif token_budget is not None and total>=token_budget:status='budget_exhausted'
        else:
            messages=[{'role':'system','content':prompt},{'role':'user','content':text}]
            assert messages[1]['content']==text
            assert not any(re.search(p,json.dumps(messages)) for _,p,_ in PATTERNS)
            response=call_with_retry(messages,t.ticket_id,retries,rng);used=response.usage.total_tokens;total+=used
            status,category=validate_output(response.choices[0].message.content)
            if status.startswith('classified_'):truth.append(t.true_category);pred.append(category)
        seg=segments.loc[t.customer_id] if t.customer_id in segments.index else None
        segment_name='' if seg is None else seg.segment_name
        if not status.startswith('classified_'):route,priority='human_review','NORMAL'
        elif category=='fraud':route,priority='security_team','HIGH'
        elif seg is not None and bool(seg.is_premium):route,priority='premium_desk','HIGH'
        else:route,priority='support_queue','NORMAL'
        rows.append(dict(ticket_id=t.ticket_id,customer_id=t.customer_id,status=status,category=category,segment_name=segment_name,priority=priority,route=route,tokens_used=used))
    result=pd.DataFrame(rows)
    summary={'status_counts':result.status.value_counts().to_dict(),'classified':len(pred),'correct':sum(a==b for a,b in zip(truth,pred)),
        'accuracy':sum(a==b for a,b in zip(truth,pred))/len(pred) if pred else 0,'total_tokens':total,'tokens_per_classified':total/len(pred) if pred else 0,
        'retries':len(retries),'redactions':redactions,'remaining_pii':0,'token_budget':token_budget}
    if output_name:
        result.to_csv(OUT/output_name,index=False)
        save_json(output_name.replace('.csv','_summary.json'),summary)
        save_json(output_name.replace('.csv','_retry_log.json'),retries)
        pd.crosstab(pd.Series(truth,name='true'),pd.Series(pred,name='predicted')).to_csv(OUT/output_name.replace('.csv','_confusion.csv'))
    return result,summary

def run_all():
    z,zs=run_pipeline(ZERO,output_name='triage_zero.csv')
    f,fs=run_pipeline(FEW,output_name='triage_few.csv')
    b,bs=run_pipeline(ZERO,6000,'triage_budget_demo.csv')
    final,final_summary=run_pipeline(FEW,100000,'triage_results.csv')
    pd.testing.assert_frame_equal(f,final)
    for s in [zs,fs]:
        assert s['status_counts']=={'classified_json':135,'classified_keyword':12,'flagged_invalid_output':9,'flagged_long':4,'skipped_empty':4}
        assert s['retries']==19 and s['redactions']=={'email':55,'phone':91,'order':132}
    assert abs(zs['accuracy']-126/147)<1e-9 and fs['accuracy']==1
    assert bs['status_counts'].get('budget_exhausted',0)>0
    save_json('part_c_summary.json',{'zero':zs,'few':fs,'budget':bs,'final':final_summary})
