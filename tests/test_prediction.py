import unittest
from unittest.mock import patch
import pandas as pd
import joblib
from predict_customer import EXAMPLE,FIELDS,score_customer,validate_field,ROOT,main

class CustomerPrediction(unittest.TestCase):
    def test_saved_pipeline_prediction_matches(self):
        model=joblib.load(ROOT/'outputs/lapse_model.joblib')
        direct=model.predict_proba(pd.DataFrame([{f:EXAMPLE[f] for f in FIELDS}]))[0,list(model.classes_).index(1)]
        result=score_customer(EXAMPLE)
        self.assertAlmostEqual(result['lapse_probability'],direct)
        self.assertEqual(result['predicted_lapsed'],int(model.predict(pd.DataFrame([EXAMPLE]))[0]))

    def test_invalid_and_cold_start_input(self):
        for field,value in [('frequency',0),('frequency',2.5),('monetary',float('inf')),('age',999),
            ('avg_discount',101),('cancellation_rate',1.1),('cancellation_rate',1),('city','Atlantis'),('avg_delivery',0)]:
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):score_customer({**EXAMPLE,field:value})
        with self.assertRaisesRegex(ValueError,'Missing'):score_customer({'city':'Chennai'})
        with self.assertRaisesRegex(ValueError,'signup'):score_customer({**EXAMPLE,'tenure':1})

    def test_inputs_normalize_and_do_not_mutate_model(self):
        path=ROOT/'outputs/lapse_model.joblib';before=path.read_bytes()
        self.assertEqual(validate_field('city',' chennai '),'Chennai')
        self.assertEqual(validate_field('favourite_category','personal care'),'Personal Care')
        score_customer(EXAMPLE)
        self.assertEqual(path.read_bytes(),before)

    def test_interactive_entry_point(self):
        import contextlib,io
        answers=['NEW001']+[str(EXAMPLE[f]) for f in FIELDS]
        output=io.StringIO()
        with patch('sys.argv',['predict_customer.py']),patch('builtins.input',side_effect=answers),contextlib.redirect_stdout(output):main()
        self.assertIn('Customer: NEW001',output.getvalue())
        self.assertIn('Estimated lapse probability:',output.getvalue())

if __name__=='__main__':unittest.main()
