#! /usr/bin/env python
# -*- coding: utf-8 -*-


print('{:^14%}'.format(0.015))
print('{:>14}'.format('陈某某'))
print('{:<14}'.format('陈某某'))
print('{:*<14}'.format('陈某某'))
print('{:&>14}'.format('陈某某'))   #填充和对齐^<>分别表示居中、左对齐、右对齐，后面带宽度


print('{:.10f}'.format(4.234324525254))
print('{:.4f}'.format(4.1))