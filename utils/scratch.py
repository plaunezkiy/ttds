expression = "(3+4)*5"
expression.replace("(", "( ")
expression.replace(")", ") ")

seq = expression.split(" ")
output = []
operator_stack = []

