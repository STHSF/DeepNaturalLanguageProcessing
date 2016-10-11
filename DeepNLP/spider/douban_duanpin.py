#!/usr/bin/python
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import re
import urllib2
import xlwt
import lxml

import sys
reload(sys)
sys.setdefaultencoding('utf8')


# 得到页面全部内容
def ask_url(url):
    request = urllib2.Request(url)  # 发送请求
    try:
        response = urllib2.urlopen(request)  # 取得响应
        html = response.read()  # 获取网页内容
        # print html
    except urllib2.URLError, e:
        if hasattr(e, "code"):
            print e.code
        if hasattr(e, "reason"):
            print e.reason
    return html


# 获取相关内容
def get_data(base_url1, base_url2):

    # 找到作者
    # pattern_author = re.compile(r'<a.+people.+">(.+)</a>')
    pattern_author = re.compile(r'<span property="v:reviewer">(.+)</span>')

    # 找到评论内容
    # pattern_subject_link = re.compile(r'<a href="(.+subject.+)" title="(.+)">')
    pattern_content = re.compile(r'<div property="v:description" class="clearfix">(.+)</div>')

    # 找到推荐等级
    # pattern_star = re.compile(r'<span class="allstar\d+" title="(.+)"></span>')
    pattern_star = re.compile(r'<span class="allstar\d+" title="(.+)"></span>')

    # 找到回应数
    # pattern_response = re.compile(r'<span class="">\((\d+)回应\)</span>')
    pattern_response = re.compile(r'<a class="pl" href=".+">\((\d+)回应\)</a>')

    # 找到有用数
    # pattern_use = re.compile(r'<em id="ucount\d+u">(\d+)</em>')
    pattern_use = re.compile(r'<span class="left">(\d+)有用.*</span>')
    # pattern_use = re.compile(r'"btn useful_count .* j a_show_login">.*(\d+).*</button>.*<button', re.DOTALL)

    remove = re.compile(r'<.+?>')  # 去除标签

    data_list = []
    for i in range(0, 2):  # 总共54页
        url = base_url1 + str(i * 20) + base_url2   # 更新url,每页有20篇文章
        html = ask_url(url)
        soup = BeautifulSoup(html)
        # 找到每一个影评项
        for item in soup.find_all('div', class_='mod-bd'):
            data = []
            item = str(item)  # 转换成字符串
            print item

            author = re.findall(pattern_author, item)
            # print author
            data.append(author)  # 添加作者

            star = re.findall(pattern_star, item)[0]
            # print star
            data.append(star)  # 添加推荐等级

            response = re.findall(pattern_response, item)
            # 回应数可能为0
            if len(response) != 0:
                response = response[0]
            else:
                response = 0
            # print response
            data.append(response)  # 添加回应数

            use = re.findall(pattern_use, item)
            # 有用数可能为0，就找不到
            if len(use) != 0:
                use = use[0]
            else:
                use = 0
            print use

            data.append(use)  # 添加有用数
            data_list.append(data)

            # 从当前网页中爬取相应数据
            review_content = re.findall(pattern_content, item)
            # print review_link
            data.append(review_content)  # 添加评论正文

    return data_list


# 将相关数据写入excel中
def save_data(data_list, save_path):
    num = len(data_list)
    print num
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet('豆瓣影评数据获取', cell_overwrite_ok=True)
    col = ('标题', '作者', '推荐级', '回应数', '有用数', '影评')
    for i in range(0, 6):
        sheet.write(0, i, col[i])  # 列名
    for i in range(0, num):  # 总共1075条影评
        data = data_list[i]
        for j in range(0, 6):
            sheet.write(i+1, j, data[j])  # 数据
    book.save(save_path)  # 保存


def main():
    base_url1 = 'https://movie.douban.com/subject/25815034/comments?start='
    base_url2 = '&limit=20&sort=new_score'
    data_list = get_data(base_url1, base_url2)
    print len(data_list)
    # save_path = u'豆瓣最受欢迎影评.xlsx'
    # save_data(data_list, save_path)

main()