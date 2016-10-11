#!/usr/bin/python
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import re
import urllib2
import xlwt
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
def get_data(base_url):

    # 找到评论原文链接
    # pattern_link = re.compile(r'<a class="".+href="(.+review.*?)"')
    pattern_link = re.compile(r'<a class="title-link" href="(.+review.*?)"')

    # 找到评论标题
    # pattern_title = re.compile(r'<a class="title".+href="(.+)"')
    # pattern_title = re.compile(r'<a href="(.+)" class="title-link"(.+)"</a>')
    pattern_title = re.compile(r'<span property="v:summary">(.+)</span>')

    # 找到作者
    # pattern_author = re.compile(r'<a.+people.+">(.+)</a>')
    pattern_author = re.compile(r'<span property="v:reviewer">(.+)</span>')

    # 找到评论的影片和影评详情链接
    # pattern_subject_link = re.compile(r'<a href="(.+subject.+)" title="(.+)">')
    pattern_subject_link = re.compile(r'<div property="v:description" class="clearfix">(.+)</div>')

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
        url = base_url + str(i * 20)  # 更新url,每页有20篇文章
        html = ask_url(url)
        soup = BeautifulSoup(html)
        # 找到每一个影评项
        for item in soup.find_all('div', class_='main review-item'):
            data = []
            item = str(item)  # 转换成字符串
            # print item

            # 从当前网页中爬取相应数据
            review_link = re.findall(pattern_link, item)
            # print review_link
            data.append(review_link)  # 添加评论正文链接

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

            # 从源网页中爬取部分数据
            for url in review_link:
                content = ask_url(url)
                content = BeautifulSoup(content)

                desc = content.find_all('div', property="v:description")
                # print desc[0]
                desc = re.sub(remove, '', str(desc))  # 去掉标签
                data.append(desc)  # 添加评论正文

                content = str(content)
                title = re.findall(pattern_title, content)[0]
                # print title
                data.append(title)  # 添加标题

                author = re.findall(pattern_author, content)
                # print author
                data.append(author)  # 添加作者

                # list_subject_link = re.findall(pattern_subject_link, content)
                # movie_name = list_subject_link[1]
                # movie_link = list_subject_link[0]
                # data.append(movie_name)  # 添加片名
                # data.append(movie_link)  # 添加影片链接

                star = re.findall(pattern_star, content)[0]
                # print star
                data.append(star)  # 添加推荐等级

    return data_list


# 将相关数据写入excel中
def save_data(data_list, save_path):
    num = len(data_list)
    print num
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet('豆瓣影评数据获取', cell_overwrite_ok=True)
    col = ('标题', '作者', '推荐级', '回应数', '影评', '有用数')
    for i in range(0, 9):
        sheet.write(0, i, col[i])  # 列名
    for i in range(0, num):  # 总共1075条影评
        data = data_list[i]
        for j in range(0, 9):
            sheet.write(i+1, j, data[j])  # 数据
    book.save(save_path)  # 保存


def main():
    # base_url='http://movie.douban.com/review/best/?start='
    # base_url = 'https://movie.douban.com/subject/25815034/reviews?start='
    base_url = 'https://movie.douban.com/subject/25815034/reviews?sort=hotest&start='
    data_list = get_data(base_url)
    print len(data_list)
    # save_path = u'豆瓣最受欢迎影评.xlsx'
    # save_data(data_list, save_path)

main()