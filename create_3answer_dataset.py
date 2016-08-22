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
    operator = ''
    if '+' in eq:
        operator = '+'
        operands = eq.split("+")
    elif '-' in eq:
        operator = '-'
        operands = eq.split("-")
    elif '*' in eq:
        operator = '*'
        operands = eq.split("*")
    elif '/' in eq:
        operator = '/'
        operands = eq.split("/")
    final_operands = []
    for op in operands:
        try:
            new = float(op)
            new2 = int(new)
            if new == new2:
                new = new2
        except ValueError:
            print "NOPE"
            new = float(op)
        final_operands.append(new)
    #operands = map(float, operands)

    txt = prob['sQuestion']
    tokens = txt.split()
    new_txt = ''
    for token in tokens:
        token = token.lower()
        token = token.strip(".")
        token = token.replace("'", '')
        token = token.strip("?")
        token = token.strip("$")
        token = token.strip(",")
        new_txt += token + " "


    print new_txt.strip(), "%s %s %s" % (str(operator), str(final_operands[0]), str(final_operands[1]))
