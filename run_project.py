"""Rebuild Parts A-D and the measured README. Run from repository root."""
from src.data_pipeline import audit,clean_data,validate,analyse
from src.customer_models import rfm_and_segments,build_lapse_model
from src.triage import run_all as triage
from src.policy_rag import run_all as rag
from src.refund_agent import run_all as agent

def main():
    print('PART A');audit();clean_data();validate();analyse()
    print('PART B');rfm_and_segments();print(build_lapse_model())
    print('PART C');triage()
    print('PART D');rag();agent()
    from src.report import write_readme
    write_readme()
    print('PROJECT COMPLETE')

if __name__=='__main__':main()
