import re
from transformers import GPT2LMHeadModel,GPT2Tokenizer
import torch
import matplotlib.pyplot as plt
import numpy as np


#Was planning on making inferences based on earnings per share and net income, but the values are not being extracted correctly.

def extract_financial_data(text):
    # Define regular expressions to match net income and EPS
    net_income_regex = r"CONSOLIDATED\s+STATEMENTS\s+OF\s+OPERATIONS.*?Net\s+income.*?(\$\s*\d[\d,.\s]*\d)" 
    eps_regex  = r"CONSOLIDATED\s+STATEMENTS\s+OF\s+OPERATIONS.*?Earnings\s+per\s+share.*?(\$\s*\d[\d,.\s]*\d)"
    
    
    # Search for net income and EPS in the text
    net_income_matches = re.findall(net_income_regex, text, re.DOTALL)
    eps_matches = re.findall(eps_regex, text, re.DOTALL)
    
    # Process net income matches
    net_income = None
    for income_match in net_income_matches:
        net_income_str = income_match  # Considering each match
        # Check if eps_str is not "1e-05"
        net_income = float(re.sub(r'[^\d.]', '', net_income_str))
        if net_income!=1e-05 and net_income >1000:
            break  # Stop searching after finding a valid EPS value
    
    # Process EPS matches
    eps = None
    for eps_match in eps_matches:
        eps_str = eps_match  # Considering each match
        # Check if eps_str is not "1e-05"
        eps = float(re.sub(r'[^\d.]', '', eps_str))
        if eps!=1e-05 and eps!=205.0 and eps >0:
            break  # Stop searching after finding a valid EPS value
    
    return net_income, eps

# Example usage
with open("sec-edgar-filings/AAPL/10-K/0001193125-14-383437/full-submission.txt", "r") as file:
    text = file.read()

net_income, eps = extract_financial_data(text)
print("Net Income:", net_income)
print("Earnings Per Share (EPS):", eps)

