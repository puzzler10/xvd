import re 
def format_sentence(x):
    """Get sentences in consistent format for evaluation. Perplexity and other fluency metrics 
    are very picky about this."""
    if len(x) == 0: return "sentence here."
    x = str(x)  # in case a number or a NaN comes through. 
    x = x.strip()  # Remove leading/trailing whitespaces
    try:
        x = x[0].upper() + x[1:]  # Ensure the first character is uppercase
        if re.search('[.!?]$', x): # Check if the sentence ends with a punctuation
            x = re.sub('\s+([.!?])$', r'\1', x)  # If it does, remove any spaces before the last punctuation mark
        else:
            x += '.'  # If it doesn't, add a full stop
    except: 
        return x
    return x
