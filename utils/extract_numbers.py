import re

def extract_numbers(s):
    # Check format validity
    if s.count('_') != 2:
        raise ValueError(f"Invalid predicate format: {s}. Expected format: 'name_t#_ed#'. Please make sure there are only two _.")
    parts = s.split('_')
    nf_name = parts[0]
    t_value = int(re.search(r't(\d+)', parts[1]).group(1))
    ed_value = int(re.search(r'ed(\d+)', parts[2]).group(1))
    return nf_name, t_value, ed_value
