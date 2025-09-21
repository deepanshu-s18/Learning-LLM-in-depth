import argparse

OPS = ['+', '-', '*', '/']

class Node():
    def __init__(self, val):
        self.val    = val
        self.left   = None
        self.right  = None


def preorder(node, prefix = ''):
    if node is None:
        return prefix
    val = node.val
    prefix += val +' '
    prefix = preorder(node.left, prefix)
    prefix = preorder(node.right, prefix)
    return prefix

def expr2tree(string):
    tokens = string.split()
    if len(tokens) == 1:
        return Node(tokens[0])
    i = 0
    while i < len(tokens):
        if tokens[i] in OPS:
            break
        i += 1

    node = Node(tokens[i])
    node.left  = expr2tree(' '.join(tokens[:i]))
    node.right = expr2tree(' '.join(tokens[i+1:])) 
    return node
