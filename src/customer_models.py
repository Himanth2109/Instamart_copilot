"""Whole-window segmentation and strictly pre-cutoff lapse modelling."""
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score,precision_recall_fscore_support,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from .common import ROOT,OUT,save_json

def load():
    return (pd.read_csv(ROOT/'data/clean/orders.csv',parse_dates=['order_ts']),
            pd.read_csv(ROOT/'data/clean/customers.csv',parse_dates=['signup_date']),
            pd.read_csv(ROOT/'data/clean/products.csv'))

def behaviour(d,snapshot):
    r=d.groupby('customer_id').agg(last=('order_ts','max'),frequency=('order_id','nunique'),monetary=('order_amount','sum'))
    r['recency']=(pd.Timestamp(snapshot)-r.pop('last')).dt.days
    return r[['recency','frequency','monetary']]

def rfm_and_segments():
    o,_,_=load(); r=behaviour(o[o.status.eq('completed')],'2025-10-01')
    assert len(r)==1175 and r.recency.median()==19 and r.frequency.max()==33
    z=StandardScaler().fit_transform(r); rows=[];models={}
    for k in range(2,9):
        m=KMeans(n_clusters=k,n_init=10,random_state=42).fit(z);models[k]=m
        rows.append({'k':k,'inertia':m.inertia_,'silhouette':silhouette_score(z,m.labels_)})
    diagnostics=pd.DataFrame(rows);diagnostics.to_csv(OUT/'kmeans_diagnostics.csv',index=False)
    ref=[1820.8,1268.2,743.1,617.4,503.2,408.5,366.5];sil=[.465,.445,.526,.481,.482,.416,.412]
    assert np.allclose(diagnostics.inertia,ref,rtol=.05) and np.allclose(diagnostics.silhouette,sil,atol=.05)
    r['segment_id']=models[4].labels_
    profile=r.groupby('segment_id').agg(size=('frequency','size'),recency=('recency','mean'),frequency=('frequency','mean'),monetary=('monetary','mean'))
    premium=int(profile.monetary.idxmax()); frequent=int(profile.frequency.idxmax()); dormant=int(profile.recency.idxmax())
    names={i:('Premium' if i==premium else 'Frequent' if i==frequent else 'Dormant' if i==dormant else 'Occasional') for i in profile.index}
    r['segment_name']=r.segment_id.map(names);r['is_premium']=r.segment_id.eq(premium)
    profile['segment_name']=profile.index.map(names)
    r.reset_index().to_csv(OUT/'customer_segments.csv',index=False);profile.to_csv(OUT/'segment_profile.csv')
    pca=PCA(n_components=2).fit(z).explained_variance_ratio_
    assert np.allclose(pca,[.634,.218],atol=.02)
    save_json('rfm_summary.json',{'summary':r[['recency','frequency','monetary']].describe().to_dict(),'pca':pca,'premium_segment':premium})
    return r,diagnostics,profile.reset_index(),pca

def build_features():
    o,c,p=load();cut=pd.Timestamp('2025-09-01');pre=o[o.order_ts.lt(cut)];done=pre[pre.status.eq('completed')]
    assert pre.order_ts.max()<cut
    x=behaviour(done,cut)
    x=x.join(done.groupby('customer_id').agg(avg_discount=('discount_pct','mean'),avg_delivery=('delivery_minutes','mean')))
    x['cancellation_rate']=pre.assign(cancelled=pre.status.eq('cancelled')).groupby('customer_id').cancelled.mean()
    cats=done.merge(p,on='product_id').groupby('customer_id').category.agg(lambda s:s.mode().iloc[0])
    x['favourite_category']=cats
    x=x.join(c.set_index('customer_id')[['city','age','gender','signup_date']])
    x['tenure']=(cut-x.pop('signup_date')).dt.days
    september=set(o.loc[o.status.eq('completed') & o.order_ts.ge(cut) & o.order_ts.lt('2025-10-01'),'customer_id'])
    y=pd.Series((~x.index.isin(september)).astype(int),index=x.index,name='lapsed')
    assert len(x)==1156 and y.sum()==443 and not x.isna().any().any()
    assert not any('segment' in col for col in x)
    return x,y

def metrics(y,pred):
    p,r,f,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
    return {'precision':p,'recall':r,'f1':f,'accuracy':accuracy_score(y,pred),'confusion_matrix':confusion_matrix(y,pred).tolist()}

def build_lapse_model():
    x,y=build_features(); xt,xv,yt,yv=train_test_split(x,y,test_size=.2,stratify=y,random_state=42)
    assert len(xt)==924 and len(xv)==232
    cat=['city','gender','favourite_category'];num=[n for n in x if n not in cat]
    def prep(): return ColumnTransformer([('num',StandardScaler(),num),('cat',OneHotEncoder(drop='first',handle_unknown='ignore'),cat)])
    lr=GridSearchCV(Pipeline([('pre',prep()),('model',LogisticRegression(max_iter=1000))]),{'model__C':[.01,.1,1,10,100]},cv=5,scoring='f1')
    rf=GridSearchCV(Pipeline([('pre',prep()),('model',RandomForestClassifier(n_estimators=100,random_state=42))]),
        {'model__max_depth':[3,5,8],'model__min_samples_leaf':[5,10,20]},cv=5,scoring='f1')
    lr.fit(xt,yt);rf.fit(xt,yt)
    # Selection uses CV only; test rows are evaluated after both searches finish.
    selected=rf if rf.best_score_>=lr.best_score_ else lr
    report={'cohort':len(x),'lapsed':int(y.sum()),'train':len(xt),'test':len(xv),'train_class_counts':yt.value_counts().to_dict(),
      'baseline':metrics(yv,np.zeros(len(yv),dtype=int)),
      'lr':{'best_params':lr.best_params_,'cv_f1':lr.best_score_,**metrics(yv,lr.predict(xv))},
      'rf':{'best_params':rf.best_params_,'cv_f1':rf.best_score_,**metrics(yv,rf.predict(xv))},'selected':'rf' if selected is rf else 'lr'}
    assert metrics(yv,selected.predict(xv))['f1']>=.55
    extra=xt.loc[yt.eq(1)].sample(n=int(yt.eq(0).sum()-yt.eq(1).sum()),replace=True,random_state=42)
    xb=pd.concat([xt,extra]);yb=pd.concat([yt,yt.loc[extra.index]])
    balanced=Pipeline([('pre',prep()),('model',LogisticRegression(C=lr.best_params_['model__C'],max_iter=1000))]).fit(xb,yb)
    report['balanced_lr']=metrics(yv,balanced.predict(xv));report['balanced_counts']=yb.value_counts().to_dict()
    report['frequency_scaler_mean']=lr.best_estimator_.named_steps['pre'].named_transformers_['num'].mean_[num.index('frequency')]
    report['frequency_cohort_mean']=x.frequency.mean()
    names=rf.best_estimator_.named_steps['pre'].get_feature_names_out()
    report['rf_top_five']=sorted(zip(names,rf.best_estimator_.named_steps['model'].feature_importances_),key=lambda t:t[1],reverse=True)[:5]
    ln=lr.best_estimator_.named_steps['pre'].get_feature_names_out();co=lr.best_estimator_.named_steps['model'].coef_[0]
    report['lr_coefficients']={n:co[list(ln).index('num__'+n)] for n in ['recency','frequency','monetary']}
    joblib.dump(selected.best_estimator_,OUT/'lapse_model.joblib');reloaded=joblib.load(OUT/'lapse_model.joblib')
    assert np.array_equal(selected.predict(xv),reloaded.predict(xv));report['reload_identical']=True
    segments=pd.read_csv(OUT/'customer_segments.csv').set_index('customer_id')
    retain=pd.DataFrame({'customer_id':xv.index,'lapse_probability':selected.predict_proba(xv)[:,1],'actually_lapsed':yv.to_numpy()})
    retain=retain[retain.customer_id.isin(segments.index[segments.is_premium])].sort_values('lapse_probability',ascending=False).head(10)
    retain.to_csv(OUT/'retention_list.csv',index=False);report['retention_true_lapsed']=int(retain.actually_lapsed.sum())
    pd.DataFrame({'customer_id':xv.index,'actually_lapsed':yv.to_numpy(),'prediction':selected.predict(xv)}).to_csv(OUT/'test_predictions.csv',index=False)
    for name,g in [('lr',lr),('rf',rf)]:
        pd.DataFrame(g.cv_results_).to_csv(OUT/f'{name}_grid_search.csv',index=False)
    save_json('part_b_summary.json',report);return report
