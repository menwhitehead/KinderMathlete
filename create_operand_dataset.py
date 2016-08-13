import json
import string

data = json.load(open("SingleOp.json", 'r'))
for prob in data:
    eq = prob['lEquations'][0]
    # print eq
    eq = eq[2:]
    # print eq
    eq = eq.strip('()')
    # print eq
    if '+' in eq:
        operands = eq.split("+")
    if '-' in eq:
        operands = eq.split("-")
    if '*' in eq:
        operands = eq.split("*")
    if '/' in eq:
        operands = eq.split("/")
    operands = map(float, operands)

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

    print new_txt.strip(), operands[0], operands[1]
    