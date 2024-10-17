import re
exp = '"middle east" AND (peace OR war)'

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
    proximity_regex = r"#(\d+)\((\w+),\s?(\w+)\)"
    phrase_regex = r"\"(.*)\""
    if re.search(proximity_regex, s):
        n, t1, t2 = re.findall(proximity_regex, s)[0]
        value = index.proximity_search(document, t1, t2, n)
    # phrase search
    elif re.search(phrase_regex, s):
        exact_query = re.findall(phrase_regex, s)[0]
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
    operator = None
    if "AND" in exp:
        operator = all
        exp = exp.split("AND")

    elif "OR" in exp:
        operator = any
        exp = exp.split("OR")
    else:
        return evaluate(exp)
    return operator(map(evaluate, exp))

evaluate_string(exp)

# def peek(a):
#     return a[-1] if a else None
# 
# def parse_string(exp):
#     """Shunting yard implementation"""
#     exp = re.sub("\(\s?", "( ", exp)
#     exp = re.sub("\s?\)", " )", exp)
#     els = exp.split(" ")
#     values, operators = [], []
#     print(els)

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
#             # while peek(operators) and peek(operators) not in "()" and greater_precedence(peek(operators), el):
#             #     operator = operators.pop()
#             #     right = values.pop()
#             #     left = values.pop()
#             #     apply_operator(operators, values)
#             values.append(el)
#     return operators, values

# print(parse_string(exp))
