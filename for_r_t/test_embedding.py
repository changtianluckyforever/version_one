#!/Users/admin/anaconda/envs/thesis/bin/python
# -*- coding: utf-8 -*-

# def small_function():
#     for i in xrange(3):
#         # count_num = count_num + 1
#         # print(count_num)
#         print(discount_val)

# # 如果全局变量和局部变量重名，当前函数输出的是局部变量，当前函数的下一级函数如果没有明确的局部变量传入，那么输出的是全局变量。
# def run_episode(discount_val=0):
#     count_num =10
#     for episode in xrange(1):
#         small_function()
#     print("^^^^^^^^")
#     print('global variable')
#     print(discount_val)
#     print(count_num)
#     print('^^^^^^^')
#
#
# def small_function():
#     for i in xrange(1):
#         # count_num = count_num + 1
#         # print(count_num)
#         print(discount_val)
#         print(count_num)
#         print('********')
# 如果全局变量和局部变量重名，当前函数输出的是局部变量，当前函数的下一级函数如果有明确的局部变量传入，那么输出的是局部变量
# discount_val = 1
# count_num = 0
# def run_episode(discount_val=0):
#     count_num =10
#     for episode in xrange(1):
#         small_function(discount_val, count_num)
#     print("^^^^^^^^")
#     print('global variable')
#     print(discount_val)
#     print(count_num)
#     print('^^^^^^^')
#
#
# def small_function(discount_val, count_num):
#     for i in xrange(1):
#         # count_num = count_num + 1
#         # print(count_num)
#         print(discount_val)
#         print(count_num)
#         print('********')
discount_val = 1
count_num = 0
from external import CountExternal
test_it = CountExternal(discount_val, count_num)

def small_function():
    for i in xrange(2):
        for j in xrange(3):
            discount_val, count_num =test_it.update_values()
            print("*********")
            print(discount_val)
            print(count_num)
            print('*********')

def run_episode():
    # count_num =10
    for episode in xrange(2):
        print('we are episode:', episode)
        small_function()
    # print("^^^^^^^^")
    # print('global variable')
    # print(discount_val)

run_episode()









