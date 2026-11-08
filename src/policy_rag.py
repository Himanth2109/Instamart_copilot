"""Word chunks, TF-IDF cosine retrieval and offline grounded answers."""
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm
from .common import ROOT,OUT,save_json

class PolicyRAG:
    def __init__(self,overlap=10):
        if not 0<=overlap<40:raise ValueError('overlap must be in [0,40)')
        self.chunks=[]
        for path in sorted((ROOT/'data/policies').glob('*.txt')):
            words=path.read_text(encoding='utf-8').split()
            for start in range(0,len(words),40-overlap):
                self.chunks.append({'source':path.name,'text':' '.join(words[start:start+40])})
                if start+40>=len(words):break
        self.vectorizer=TfidfVectorizer();self.matrix=self.vectorizer.fit_transform([c['text'] for c in self.chunks])

    def retrieve(self,query,k=3):
        scores=cosine_similarity(self.vectorizer.transform([query]),self.matrix)[0]
        order=scores.argsort()[::-1][:k]
        return [dict(self.chunks[i],score=float(scores[i])) for i in order]

    def answer(self,question):
        chunks=self.retrieve(question)
        context='\n'.join(f"[{c['source']}]\n{c['text']}" for c in chunks)
        messages=[{'role':'system','content':'POLICY ASSISTANT: answer only from supplied context. If absent, say I could not find that information in the policy documents.'},
          {'role':'user','content':f'CONTEXT:\n{context}\nQUESTION:\n{question}'}]
        r=mock_llm.chat(messages)
        return {'answer':r.choices[0].message.content,'sources':[c['source'] for c in chunks],'tokens_used':r.usage.total_tokens}

def evaluate(rag):
    rows=[]
    for q in pd.read_csv(ROOT/'data/raw/policy_eval.csv',keep_default_na=False).itertuples(index=False):
        result=rag.answer(q.question);covered=q.expected_doc!='NONE'
        rows.append({'question':q.question,'expected_doc':q.expected_doc,'expected_fact':q.expected_fact,**result,
            'retrieval_hit':covered and q.expected_doc in result['sources'],
            'fact_hit':covered and q.expected_fact in result['answer'],
            'fallback':not covered and result['answer'].startswith('I could not find')})
    return pd.DataFrame(rows)

def run_all():
    summaries={}
    for overlap in [10,0]:
        rag=PolicyRAG(overlap);d=evaluate(rag);d.to_csv(OUT/f'policy_eval_overlap_{overlap}.csv',index=False)
        summaries[str(overlap)]={'chunks':len(rag.chunks),'by_source':dict(Counter(c['source'] for c in rag.chunks)),
          'retrieval_hits':int(d.retrieval_hit.sum()),'fact_hits':int(d.fact_hit.sum()),'fallbacks':int(d.fallback.sum()),
          'failed_questions':d.loc[d.expected_doc.ne('NONE') & ~d.fact_hit,'question'].tolist()}
        if overlap==10:
            assert summaries['10']['chunks']==15
            assert (d.retrieval_hit.sum(),d.fact_hit.sum(),d.fallback.sum())==(12,12,2)
            assert all(c['source']=='refund_policy.txt' for c in rag.retrieve('refund policy late delivery'))
        else:assert d.fact_hit.sum()<12
    save_json('part_d_rag_summary.json',summaries)
