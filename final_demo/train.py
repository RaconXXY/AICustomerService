#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Date    : 2019-01-16 13:04:42
# @Author  : Wang Yinkai (15057638632@163.com)
# @Link    : http://xjwyk.top
# @Version : $Id$

import os

os.system("python util.py")
os.system("python data_prepare.py")
os.system("python extract_feature.py")
os.system("python main.py")

