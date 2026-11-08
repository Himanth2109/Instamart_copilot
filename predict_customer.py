"""Enter a customer's historical behaviour and score with the saved lapse model."""
import argparse
import math
from pathlib import Path
import joblib
import pandas as pd

ROOT=Path(__file__).resolve().parent
FIELDS=['recency','frequency','monetary','avg_discount','avg_delivery','cancellation_rate','favourite_category','city','age','gender','tenure']
CHOICES={
    'city':['Chennai','Mumbai','Bengaluru','Delhi','Hyderabad','Pune'],
    'gender':['male','female','unknown'],
    'favourite_category':['Grocery','Dairy','Snacks','Personal Care','Electronics']}
LABELS={
    'recency':'Whole days since the last completed order',
    'frequency':'Number of completed orders in the history window',
    'monetary':'Total completed-order spend in INR in that window',
    'avg_discount':'Average discount percentage on completed orders (0 to 100)',
    'avg_delivery':'Average completed-order delivery time in minutes',
    'cancellation_rate':'Cancelled orders divided by all final orders (0 to 1)',
    'favourite_category':'Most frequently ordered product category',
    'city':'Customer city',
    'age':'Age in years',
    'gender':'Gender',
    'tenure':'Whole days from signup to the scoring date'}
EXAMPLE=dict(customer_id='DEMO_CUSTOMER',recency=20,frequency=6,monetary=1800,avg_discount=10,
    avg_delivery=28,cancellation_rate=.1,favourite_category='Grocery',city='Chennai',age=30,gender='female',tenure=300)

def validate_field(name,value):
    if name in CHOICES:
        text=str(value).strip()
        match=next((v for v in CHOICES[name] if v.lower()==text.lower()),None)
        if match is None:raise ValueError('Choose one of: '+', '.join(CHOICES[name]))
        return match
    try:number=float(value)
    except (TypeError,ValueError):raise ValueError('Enter a number.') from None
    if not math.isfinite(number):raise ValueError('Enter a finite number.')
    if number<0:raise ValueError('Enter zero or a positive number.')
    if name in ['recency','frequency','tenure'] and not number.is_integer():raise ValueError('Enter a whole number.')
    if name=='frequency' and number<1:raise ValueError('At least one completed order is required; never-ordered customers cannot be scored by this model.')
    if name=='age' and not 10<=number<=100:raise ValueError('Age must be between 10 and 100.')
    if name=='avg_discount' and number>100:raise ValueError('Discount must be between 0 and 100.')
    if name=='cancellation_rate' and number>1:raise ValueError('Use a fraction from 0 to 1; for 10 percent enter 0.1.')
    if name=='cancellation_rate' and number==1:raise ValueError('A customer with completed orders cannot have all final orders cancelled.')
    if name=='avg_delivery' and number<=0:raise ValueError('Delivery time must be greater than zero.')
    return int(number) if name in ['recency','frequency','tenure'] else number

def score_customer(details,model_path=None):
    missing=[f for f in FIELDS if f not in details]
    if missing:raise ValueError('Missing required details: '+', '.join(missing))
    row={f:validate_field(f,details[f]) for f in FIELDS}
    if row['recency']>row['tenure']:raise ValueError('Days since last order cannot exceed days since signup.')
    path=Path(model_path) if model_path else ROOT/'outputs/lapse_model.joblib'
    if not path.is_file():raise ValueError('Saved model not found. Run run_project.py first.')
    model=joblib.load(path)
    data=pd.DataFrame([row])
    positive=list(model.classes_).index(1)
    probability=float(model.predict_proba(data)[0,positive]);label=int(model.predict(data)[0])
    return {'customer_id':str(details.get('customer_id','NEW_CUSTOMER')),'lapse_probability':probability,
        'predicted_lapsed':label,'prediction':'LIKELY TO LAPSE' if label else 'LIKELY TO REMAIN ACTIVE'}

def interactive():
    print('\nCustomer lapse prediction — educational demo')
    print('Use the five months of order history ending BEFORE the scoring month, as in training.')
    print('Lapse means no completed orders during the following month.')
    print('Age/city alone are insufficient. A never-ordered customer cannot be scored.')
    print('Use synthetic customer identifiers; no name, phone number or email is needed.\n')
    details={'customer_id':input('Customer identifier [NEW_CUSTOMER]: ').strip() or 'NEW_CUSTOMER'}
    for field in FIELDS:
        while True:
            label=LABELS[field]
            if field in CHOICES:label+=' ('+', '.join(CHOICES[field])+')'
            try:details[field]=validate_field(field,input(label+': '))
            except ValueError as e:print('Please try again:',e);continue
            if field=='tenure' and details['tenure']<details['recency']:
                print('Days since signup must be at least days since last order.');continue
            break
    return details

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--example',action='store_true',help='Score a clearly labelled fictional example without typing inputs.')
    args=parser.parse_args()
    try:
        details=EXAMPLE if args.example else interactive()
        if args.example:print('Fictional example inputs:\n'+pd.Series(details).to_string())
        result=score_customer(details)
    except (EOFError,KeyboardInterrupt):print('\nCancelled. No customer record was changed.');return
    except ValueError as e:parser.exit(1,'Unable to score: '+str(e)+'\n')
    print('\nCustomer:',result['customer_id'])
    print(f"Estimated lapse probability: {result['lapse_probability']:.1%}")
    print('Prediction:',result['prediction'])
    print('Decision threshold: 50% (class 1 means lapsed).')
    print('This is a prediction from a synthetic-data model, not an observed outcome or a validated production probability.')
    print('No training data, customer database or model was changed.')

if __name__=='__main__':main()
