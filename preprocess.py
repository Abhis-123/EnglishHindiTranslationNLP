import string 
import re

def preprocess(text, add_token=True):
  text = ''.join(ch for ch in text if ch not in string.punctuation)
  text = text.lower()
  text = re.sub(r'\d','',text)
  text = re.sub(r'\s+',' ',text)
  text = text.strip()
  if add_token:
    text = "<START> " + text + " <END>" 
  return text