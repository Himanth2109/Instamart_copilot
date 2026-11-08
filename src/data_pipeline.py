"""Raw audit, documented cleaning, SQLite analysis and exact checkpoints."""
import sqlite3
import numpy as np
import pandas as pd
from .common import ROOT, OUT, save_json

def audit():
    tables = {n:pd.read_csv(ROOT/f'data/raw/{n}.csv',keep_default_na=False) for n in ['customers','products','orders']}
    report = {}
    for n,d in tables.items():
        missing=d.eq('')
        report[n]={'shape':list(d.shape),'dtypes':d.dtypes.astype(str).to_dict(),
                   'missing_count':missing.sum().to_dict(),'missing_pct':(100*missing.mean()).to_dict()}
    o,c=tables['orders'],tables['customers']
    ages=pd.to_numeric(c.age,errors='coerce'); delivery=pd.to_numeric(o.delivery_minutes,errors='coerce')
    report['specific']={'raw_rows':len(o),'exact_duplicates':int(o.duplicated().sum()),'pending':int(o.status.eq('pending').sum()),
      'distinct_orders':o.order_id.nunique(),'non_numeric_amounts':int(pd.to_numeric(o.order_amount,errors='coerce').isna().sum()),
      'negative_delivery':int(delivery.lt(0).sum()),'blank_delivery':int(o.delivery_minutes.eq('').sum()),
      'blank_age':int(c.age.eq('').sum()),'invalid_age':int((ages.notna() & ~ages.between(10,100)).sum()),
      'city_spellings':c.city.nunique(),'gender_spellings':c.loc[c.gender.ne(''),'gender'].nunique(),'blank_gender':int(c.gender.eq('').sum())}
    save_json('audit.json',report)
    return report

def clean_data():
    c=pd.read_csv(ROOT/'data/raw/customers.csv'); p=pd.read_csv(ROOT/'data/raw/products.csv'); o=pd.read_csv(ROOT/'data/raw/orders.csv')
    changes={'city_values_changed':int(c.city.ne(c.city.str.strip().str.title()).sum())}
    c.city=c.city.str.strip().str.title()
    old=c.gender.copy(); c.gender=c.gender.fillna('unknown').str.lower().replace({'m':'male','f':'female'})
    changes['gender_values_changed']=int(old.fillna('').ne(c.gender).sum())
    c.age=pd.to_numeric(c.age,errors='coerce').where(lambda x:x.between(10,100))
    changes['ages_imputed']=int(c.age.isna().sum()); c.age=c.age.fillna(c.groupby('city').age.transform('median'))
    c.signup_date=pd.to_datetime(c.signup_date); o.order_ts=pd.to_datetime(o.order_ts)
    changes['exact_rows_removed']=int(o.duplicated().sum()); o=o.drop_duplicates()
    n=len(o); o=o.sort_values('order_ts').drop_duplicates('order_id',keep='last').sort_values('order_id').reset_index(drop=True)
    changes['earlier_records_removed']=n-len(o)
    o.order_amount=pd.to_numeric(o.order_amount,errors='coerce'); changes['amounts_reconstructed']=int(o.order_amount.isna().sum())
    o.order_amount=o.order_amount.fillna((o.product_id.map(p.set_index('product_id').price_inr)*o.quantity*(1-o.discount_pct/100)).round(2))
    changes['negative_durations_cleared']=int(o.delivery_minutes.lt(0).sum()); o.delivery_minutes=o.delivery_minutes.mask(o.delivery_minutes.lt(0))
    completed=o.status.eq('completed'); category=o.product_id.map(p.set_index('product_id').category)
    changes['completed_durations_imputed']=int(o.loc[completed,'delivery_minutes'].isna().sum())
    o.loc[completed,'delivery_minutes']=o.loc[completed,'delivery_minutes'].fillna(o.loc[completed].groupby(category[completed]).delivery_minutes.transform('median'))
    (ROOT/'data/clean').mkdir(exist_ok=True)
    with sqlite3.connect(ROOT/'data/instamart.db') as con:
        for name,d in [('customers',c),('products',p),('orders',o)]:
            d.to_csv(ROOT/f'data/clean/{name}.csv',index=False)
            d.to_sql(name,con,if_exists='replace',index=False)
        con.execute('CREATE UNIQUE INDEX orders_id ON orders(order_id)')
    save_json('cleaning_changes.json',changes)
    return c,p,o

def validate():
    c=pd.read_csv(ROOT/'data/clean/customers.csv');o=pd.read_csv(ROOT/'data/clean/orders.csv');done=o[o.status.eq('completed')]
    assert len(c)==1200 and c.customer_id.is_unique
    assert c.city.value_counts().to_dict()=={'Pune':227,'Delhi':210,'Bengaluru':197,'Chennai':191,'Hyderabad':191,'Mumbai':184}
    assert c.gender.value_counts().to_dict()=={'female':610,'male':560,'unknown':30}
    assert c.age.notna().all() and c.age.between(18,60).all()
    assert len(o)==12309 and o.order_id.is_unique
    assert o.status.value_counts().to_dict()=={'completed':11110,'cancelled':1199}
    assert o.order_amount.notna().all()
    assert done.delivery_minutes.notna().all() and done.delivery_minutes.ge(0).all()
    assert o.loc[o.status.eq('cancelled'),'delivery_minutes'].isna().sum()==1199
    assert np.isclose(done.order_amount.sum(),2998668.80)
    print('ALL CHECKPOINTS PASSED')

def analyse():
    with sqlite3.connect(ROOT/'data/instamart.db') as con:
        queries=[q.strip() for q in (ROOT/'sql/queries.sql').read_text().split(';') if q.strip()]
        results=[pd.read_sql_query(q,con) for q in queries]
    for i,d in enumerate(results,1):d.to_csv(OUT/f'query_{i}.csv',index=False)
    assert [len(d) for d in results]==[30,30,11,25,550,1175,183]
    assert np.isclose(results[1].revenue.sum(),2998668.8)
    assert np.isclose(results[6].cumulative_revenue.iloc[-1],2998668.8)
    assert (results[5]['rank']!=results[5].row_number).sum()==8
    c=pd.read_csv(ROOT/'data/clean/customers.csv');p=pd.read_csv(ROOT/'data/clean/products.csv');o=pd.read_csv(ROOT/'data/clean/orders.csv')
    d=o[o.status.eq('completed')].merge(c,on='customer_id').merge(p,on='product_id')
    pivot=pd.pivot_table(d,values='order_amount',index='city',columns='category',aggfunc='sum',margins=True)
    assert np.isclose(pivot.loc['All','All'],results[1].revenue.sum());pivot.to_csv(OUT/'city_category_pivot.csv')
    city=d.groupby('city').agg(sum=('order_amount','sum'),mean=('order_amount','mean'),count=('order_amount','count'));city.to_csv(OUT/'city_summary.csv')
    a=d.order_amount;q1,q3=a.quantile([.25,.75]);low,high=q1-1.5*(q3-q1),q3+1.5*(q3-q1)
    corr=d[['order_amount','discount_pct','quantity','delivery_minutes']].corr();corr.to_csv(OUT/'correlations.csv')
    report={'mean':a.mean(),'median':a.median(),'skewness':a.skew(),'q1':q1,'q3':q3,'lower_fence':low,'upper_fence':high,
      'below_fence':int(a.lt(low).sum()),'above_fence':int(a.gt(high).sum()),'q5_average':results[4].average_spend.iloc[0],
      'top_customers':results[5].query('row_number==1').to_dict('records')}
    save_json('part_a_summary.json',report)
    return report
