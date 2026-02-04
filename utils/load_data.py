import os
import pandas as pd
from pathlib import Path

PROJECT_HOME = Path(__file__).parent.parent.resolve()
DATA_DIR = os.path.join(PROJECT_HOME, 'data')
TEMPLATE_DIR = os.path.join(PROJECT_HOME, 'templates')
PRIVASIS_DIR = os.path.join(PROJECT_HOME, 'outputs', 'privasis')
FILTERED_PRIVASIS_DIR = os.path.join(PROJECT_HOME, 'outputs', 'filtered_privasis')

def load_jsonl(filepath):
    return pd.read_json(filepath, lines=True)

def load_names():
    name_file_path = os.path.join(DATA_DIR, "names", "name_frequency.jsonl")
    names = pd.read_json(name_file_path, lines=True)
    names['F'] = names['F'].astype(int)
    names['M'] = names['M'].astype(int)
    return names

# def load_profile_base_template(gen_w_event):
#     if gen_w_event:
#         template_path = os.path.join(TEMPLATE_DIR, "profile", "template_profile_v0.5.txt")
#     else:
#         template_path = os.path.join(TEMPLATE_DIR, "profile", "template_profile_v0.txt")

def load_countries():
    country_file_path = os.path.join(DATA_DIR, "countries", "countries_list.txt")
    # Open the text file for reading
    with open(country_file_path, 'r') as file:
        # Read the lines and store them in a list
        countries = file.readlines()

    return countries

def load_religions():
    religion_file_path = os.path.join(DATA_DIR, "religions", "religion_list.txt")
    # Open the text file for reading
    with open(religion_file_path, 'r') as file:
        # Read the lines and store them in a list
        religions = file.readlines()

    return religions

def load_profile(path="profile/template_profile_v0.txt"):
    if path is None:
        return None
    
    template_path = os.path.join(TEMPLATE_DIR, path)
    with open(template_path, 'r') as file:
        profile_template = file.read()
    return profile_template

def load_privasis(id):
    privasis_file_path = os.path.join(PRIVASIS_DIR, f"{id}.jsonl")
    privasis = pd.read_json(privasis_file_path, lines=True)
    return privasis

def load_filtered_privasis(filename):
    privasis_file_path = os.path.join(FILTERED_PRIVASIS_DIR, f"{filename}.jsonl")
    privasis = pd.read_json(privasis_file_path, lines=True)
    return privasis