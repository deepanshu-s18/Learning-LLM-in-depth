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
