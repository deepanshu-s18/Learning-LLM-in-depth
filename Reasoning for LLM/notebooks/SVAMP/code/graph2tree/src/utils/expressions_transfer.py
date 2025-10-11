from copy import deepcopy
import re


# An expression tree node
class Et:
    # Constructor to create a node
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


# Returns root of constructed tree for given postfix expression
def construct_exp_tree(postfix):
    stack = []

    # Traverse through every character of input expression
    for char in postfix:

        # if operand, simply push into stack
