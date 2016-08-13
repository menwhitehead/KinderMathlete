import json
import string

data = json.load(open("SingleOp.json", 'r'))
for prob in data:
    eq = prob['lEquations'][0]
    txt = prob['sQuestion']
    tokens = txt.split()
    new_txt = ''
    for token in tokens:
        token = token.lower()
        token = token.strip(".")
        token = token.replace("'", '')
        token = token.strip("?")
        token = token.strip("$")
        new_txt += token + " "

    print new_txt.strip(), 
    #print eq,
    if '+' in eq: print "ADDITION"
    if '-' in eq: print "SUBTRACTION"
    if '*' in eq: print "MULTIPLICATION"
    if '/' in eq: print "DIVISION"