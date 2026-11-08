"""Bounded ReAct runner with persisted observations and idempotent ledger writes."""
import json
import math
import sqlite3
import time
import uuid
from datetime import datetime,timezone
import pandas as pd
import mock_llm
from .common import ROOT,OUT,save_json
from .policy_rag import PolicyRAG

STATE=ROOT/'state';STATE.mkdir(exist_ok=True)
_rag=None
_execution_key=None

def get_order(order_id):
    with sqlite3.connect(ROOT/'data/instamart.db') as con:
        con.row_factory=sqlite3.Row
        r=con.execute('SELECT order_id,status,order_amount,delivery_minutes FROM orders WHERE order_id=?',(order_id,)).fetchone()
    return dict(r) if r else {'error':'order not found'}

def search_policy(query):
    global _rag
    if _rag is None:_rag=PolicyRAG()
    return {'results':_rag.retrieve(query)}

def calculate_refund(order_amount,delivery_minutes):
    amount=float(order_amount)
    if not math.isfinite(amount) or amount<0:raise ValueError('invalid amount')
    # Status eligibility is independently enforced by the issuing tools.
    return {'refund_inr':round(amount,2) if delivery_minutes is not None and float(delivery_minutes)>45 else 0.0}

def _ledger(tool,order_id,refund_inr):
    order=get_order(order_id)
    if 'error' in order or order['status']!='completed':raise ValueError('only completed orders can be refunded')
    expected=calculate_refund(order['order_amount'],order['delivery_minutes'])['refund_inr']
    if expected<=0 or abs(float(refund_inr)-expected)>.001:raise ValueError('refund must equal eligible full amount')
    if tool=='issue_refund' and expected>1000:raise ValueError('human approval required')
    if tool=='escalate_to_human' and expected<=1000:raise ValueError('automatic refund threshold not exceeded')
    path=OUT/'refund_ledger.jsonl'
    # A stable execution key survives a crash between ledger append and checkpoint.
    if path.exists() and _execution_key:
        for line in path.read_text(encoding='utf-8').splitlines():
            prior=json.loads(line)
            if prior.get('execution_key')==_execution_key:return {'refund_inr':prior['refund_inr']}
    record={'timestamp':datetime.now(timezone.utc).isoformat(),'tool':tool,'order_id':order_id,'refund_inr':expected,'execution_key':_execution_key}
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
    return {'refund_inr':expected}

def issue_refund(order_id,refund_inr):return _ledger('issue_refund',order_id,refund_inr)
def escalate_to_human(order_id,refund_inr):return _ledger('escalate_to_human',order_id,refund_inr)

TOOLS_SPEC=[
 {'name':'get_order','description':'Look up a final order in the clean SQLite database.','parameters':{'type':'object','properties':{'order_id':{'type':'string'}},'required':['order_id'],'additionalProperties':False}},
 {'name':'search_policy','description':'Retrieve the three most similar policy chunks.','parameters':{'type':'object','properties':{'query':{'type':'string'}},'required':['query'],'additionalProperties':False}},
 {'name':'calculate_refund','description':'Calculate the full amount for delivery over 45 minutes, otherwise zero.','parameters':{'type':'object','properties':{'order_amount':{'type':'number'},'delivery_minutes':{'type':['number','null']}},'required':['order_amount','delivery_minutes'],'additionalProperties':False}},
 {'name':'issue_refund','description':'Record an eligible automatic refund up to INR 1000.','parameters':{'type':'object','properties':{'order_id':{'type':'string'},'refund_inr':{'type':'number'}},'required':['order_id','refund_inr'],'additionalProperties':False}},
 {'name':'escalate_to_human','description':'Record a refund above INR 1000 for human approval without issuing it.','parameters':{'type':'object','properties':{'order_id':{'type':'string'},'refund_inr':{'type':'number'}},'required':['order_id','refund_inr'],'additionalProperties':False}}
]
TOOL_MAP={f.__name__:f for f in [get_order,search_policy,calculate_refund,issue_refund,escalate_to_human]}

class SimulatedCrash(RuntimeError):pass

def _checkpoint(s):
    path=STATE/(s['run_id']+'.json');temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(s,indent=2),encoding='utf-8');temp.replace(path)

def _finish(s,status,answer):
    s['status']=status;s['final_answer']=answer;_checkpoint(s)
    result={k:s[k] for k in ['run_id','scenario_id','status','final_answer','steps','llm_calls','tokens_used']}
    with (OUT/'agent_runs.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(result)+'\n')
    return result

def _continue(s,crash_after_tools=None):
    global _execution_key
    start=time.monotonic()
    def fallback(kind):return _finish(s,'FALLBACK_'+kind,f"Human handoff: {kind.lower().replace('_',' ')}; scenario {s['scenario_id']}; completed tools: "+', '.join(s['tools_executed']))
    while s['llm_calls']<s['max_iterations']:
        if time.monotonic()-start>=s['time_limit_seconds']:return fallback('TIMEOUT')
        response=mock_llm.chat(s['messages'],tools=TOOLS_SPEC)
        s['llm_calls']+=1;s['tokens_used']+=response.usage.total_tokens
        if s['tokens_used']>s['token_limit']:return fallback('TOKEN_LIMIT')
        msg=response.choices[0].message
        if not msg.tool_calls:
            text=msg.content or '';status=text.split(':',1)[0]
            return _finish(s,status,text)
        calls=[{'id':t.id,'name':t.name,'arguments':t.arguments} for t in msg.tool_calls]
        s['messages'].append({'role':'assistant','content':None,'tool_calls':calls})
        for call in calls:
            name,args=call['name'],call['arguments']
            if name not in TOOL_MAP:return fallback('INVALID_TOOL')
            spec=next(t for t in TOOLS_SPEC if t['name']==name)['parameters']
            if not isinstance(args,dict) or set(args)!=set(spec['required']):return fallback('INVALID_TOOL_ARGUMENTS')
            _execution_key=f"{s['run_id']}:{len(s['steps'])+1}"
            try:observation=TOOL_MAP[name](**args)
            except (ValueError,TypeError) as e:return _finish(s,'FALLBACK_TOOL_ERROR','Human handoff: '+str(e))
            finally:_execution_key=None
            s['messages'].append({'role':'tool','tool_call_id':call['id'],'name':name,'content':json.dumps(observation)})
            s['steps'].append({'tool':name,'arguments':args,'observation':observation})
            s['tools_executed'].append(name);s['step_number']=len(s['steps']);_checkpoint(s)
            if crash_after_tools==len(s['steps']):raise SimulatedCrash(s['run_id'])
    return fallback('MAX_ITERATIONS')

def run_agent(user_message,scenario_id,max_iterations=6,token_limit=6000,time_limit_seconds=10,run_id=None,crash_after_tools=None):
    run_id=run_id or uuid.uuid4().hex
    if not run_id.replace('-','').replace('_','').isalnum():raise ValueError('invalid run id')
    s={'run_id':run_id,'scenario_id':scenario_id,'messages':[{'role':'system','content':'AGENT: use tools to verify refund eligibility; all requests are within seven days. Never issue above INR 1000.'},{'role':'user','content':user_message}],
      'tokens_used':0,'llm_calls':0,'step_number':0,'steps':[],'tools_executed':[],'max_iterations':max_iterations,'token_limit':token_limit,'time_limit_seconds':time_limit_seconds}
    if (STATE/(run_id+'.json')).exists():raise ValueError('run id already exists; use resume_agent')
    return _continue(s,crash_after_tools)

def resume_agent(run_id):
    if not run_id.replace('-','').replace('_','').isalnum():raise ValueError('invalid run id')
    s=json.loads((STATE/(run_id+'.json')).read_text(encoding='utf-8'))
    if 'status' in s:return {k:s[k] for k in ['run_id','scenario_id','status','final_answer','steps','llm_calls','tokens_used']}
    return _continue(s)

def run_all():
    # Only generated files belonging to this demonstration are reset.
    for name in ['agent_runs.jsonl','refund_ledger.jsonl']:(OUT/name).write_text('',encoding='utf-8')
    for path in STATE.glob('demo_*.json'):path.unlink()
    scenarios=pd.read_csv(ROOT/'data/raw/agent_scenarios.csv',keep_default_na=False);results=[]
    for t in scenarios.itertuples(index=False):
        r=run_agent(t.message,t.scenario_id,run_id='demo_'+t.scenario_id)
        assert r['status']==('FALLBACK_MAX_ITERATIONS' if t.scenario_id=='S7' else t.expected_outcome)
        refund=next((step['observation']['refund_inr'] for step in r['steps'] if step['tool'] in ['issue_refund','escalate_to_human']),0)
        assert abs(refund-float(t.expected_refund_inr))<.01
        results.append({**{k:r[k] for k in ['scenario_id','status','llm_calls','tokens_used']},'refund_inr':refund})
    assert results[0]['llm_calls']==5 and results[-1]['llm_calls']==6
    s1=scenarios.iloc[0]
    token=run_agent(s1.message,'S1_token_demo',token_limit=200,run_id='demo_token')
    timeout=run_agent(s1.message,'S1_timeout_demo',time_limit_seconds=0,run_id='demo_timeout')
    assert token['status']=='FALLBACK_TOKEN_LIMIT' and timeout['status']=='FALLBACK_TIMEOUT' and timeout['llm_calls']==0
    try:run_agent(s1.message,'S1_resume_demo',run_id='demo_crash',crash_after_tools=2)
    except SimulatedCrash:pass
    before=json.loads((STATE/'demo_crash.json').read_text())
    assert before['tools_executed']==['get_order','search_policy']
    resumed=resume_agent('demo_crash')
    assert resumed['status']=='REFUND'
    tools=[s['tool'] for s in resumed['steps']]
    assert tools==['get_order','search_policy','calculate_refund','issue_refund']
    assert resumed['steps'][-1]['observation']['refund_inr']==128.25
    save_json('part_d_agent_summary.json',{'scenarios':results,'token_demo':token['status'],'timeout_demo':timeout['status'],'resume_tools':tools,'resumed_status':resumed['status']})
