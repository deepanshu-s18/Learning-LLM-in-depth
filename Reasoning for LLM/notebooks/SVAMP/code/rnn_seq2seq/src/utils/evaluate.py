from sympy import Eq, solve
from sympy.parsing.sympy_parser import parse_expr
import sympy as sp
import pdb

"""
EXAMPLE:
prefix1 = '* n0 + + n1 n2 n3'
list_num1 = [13, 9, 10, 3] n0->13, n1->9, n2->10, n3->3
print(ans_evaluator(prefix1, list_num1))
answer: 286
"""

def format_eq(eq):
	fin_eq = ""
	ls = ['0','1','2','3','4','5','6','7','8','9','.']
	temp_num = ""
	flag = 0
	for i in eq:
		if flag > 0:
			fin_eq = fin_eq + i
			flag = flag-1
		elif i == 'n':
			flag = 6
