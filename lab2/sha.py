import re
exp = '"middle east" AND peace'

operators = {
    "AND": all,
    "OR": any,
    "NOT": lambda e: not e,
    "": ""
}

def is_operator(q):
    return q in operators

def evaluate(s):
    negated = False
    value = False
    if "NOT" in s:
        negated = True
        s = s.split("NOT")[1]
    # proximity search
    if re.extract("#n(t1,t2)") as n, t1, t2:
        value = index.proximity_search(document, t1, t2, n)
    # phrase search
    elif re.extract('\"exact_string\"') as exact_string:
        value = index.exact_match(document, exact_string)
        # OR
        # proximity search n=1
        value = index.proximity_search(document, t1, t2, 1)
    # regular search
    else:
        value = index.contains(document, s)
    # 
    return not value if negated else value

def evaluate_string(exp):
    if "AND" in exp:
        exp.split("AND")

        all(
            map(evaluate, exp)
        )

    elif "OR" in exp:
        exp.split("OR")
    return

evaluate_string(exp)

# def evaluate_string(exp):
#    """Shunting yard implementation"""
#     exp.replace("(_?", "( ")
#     exp.replace("_?)", " )")
#     els = exp.split(" ")
#     values, operators = [], []

#     for el in els:
#         if is_operator(el):
#             operators.append(el)
#         elif el == "(":
#             operators.append(el)
#         elif el == ")":
#             while peek(operators) != "(":
#                 # evaluate stack
#                 continue
#             operators.pop()
#         else:
#             while peek(operators) and peek(operators) not in "()" and greater_precedence(peek(operators), el):
#                 operator = operators.pop()
#                 right = values.pop()
#                 left = values.pop()
#                 apply_operator(operators, values)
#             values.append(el)
#     return
