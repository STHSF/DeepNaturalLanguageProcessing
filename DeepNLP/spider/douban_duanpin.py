#!/usr/bin/python
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import re
import urllib2
import xlwt
import lxml
import time

import sys
reload(sys)
sys.setdefaultencoding('utf8')


# 得到页面全部内容
def url_request(url):

    # 发送带cookie的请求，豆瓣的cookie
    # opener = urllib2.build_opener()
    # opener.addheaders.append(('Cookie', 'bid="pkYMjawrS3w"; ll="108296"; ps=y; as="https://movie.douban.com/subject/25731059/comments"; ap=1; ct=y; _pk_ref.100001.4cf6=%5B%22%22%2C%22%22%2C1476238834%2C%22https%3A%2F%2Fwww.baidu.com%2Fs%3Fie%3DUTF-8%26tn%3Dnull%26wd%3D%25E8%25B1%2586%25E7%2593%25A3%22%5D; _pk_id.100001.4cf6=a3a1f5c47b5ff626.1476007294.12.1476238839.1476236448.; _pk_ses.100001.4cf6=*; __utma=30149280.696462676.1453800947.1476235933.1476238834.32; __utmb=30149280.0.10.1476238834; __utmc=30149280; __utmz=30149280.1476007274.20.20.utmcsr=baidu|utmccn=(organic)|utmcmd=organic; __utma=223695111.769361968.1476007295.1476235933.1476238834.12; __utmb=223695111.0.10.1476238834; __utmc=223695111; __utmz=223695111.1476238834.12.2.utmcsr=baidu|utmccn=(organic)|utmcmd=organic|utmctr=%E8%B1%86%E7%93%A3; _vwo_uuid_v2=97F16FB3676F5D3A2122B671FA6C1077|d2642bd65fbb128b7a96444dd9130990'))
    # 添加useragent
    request = urllib2.Request(url)  # 发送请求
    request.add_header('User-Agent',
                       'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36')
    request.add_header('Host', "movie.douban.com")
    request.add_header('Cookie', 'bid="pkYMjawrS3w"; ll="108296"; ps=y; as="https://movie.douban.com/subject/25731059/comments"; ap=1; ct=y; _pk_ref.100001.4cf6=%5B%22%22%2C%22%22%2C1476238834%2C%22https%3A%2F%2Fwww.baidu.com%2Fs%3Fie%3DUTF-8%26tn%3Dnull%26wd%3D%25E8%25B1%2586%25E7%2593%25A3%22%5D; _pk_id.100001.4cf6=a3a1f5c47b5ff626.1476007294.12.1476238839.1476236448.; _pk_ses.100001.4cf6=*; __utma=30149280.696462676.1453800947.1476235933.1476238834.32; __utmb=30149280.0.10.1476238834; __utmc=30149280; __utmz=30149280.1476007274.20.20.utmcsr=baidu|utmccn=(organic)|utmcmd=organic; __utma=223695111.769361968.1476007295.1476235933.1476238834.12; __utmb=223695111.0.10.1476238834; __utmc=223695111; __utmz=223695111.1476238834.12.2.utmcsr=baidu|utmccn=(organic)|utmcmd=organic|utmctr=%E8%B1%86%E7%93%A3; _vwo_uuid_v2=97F16FB3676F5D3A2122B671FA6C1077|d2642bd65fbb128b7a96444dd9130990')

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
    pattern_author = re.compile(r'<a.+people.+">(.+)</a>')

    # 找到评论内容
    pattern_content = re.compile(r'<p class=""> (.+).*</p>', re.DOTALL)

    # 找到推荐等级
    pattern_star = re.compile(r'<span class="allstar(\d+) rating" title=".*"></span>')

    # 找到有用数
    pattern_use = re.compile(r'<span class="votes pr5">(\d+).*</span>', re.DOTALL)

    remove = re.compile(r'<.+?>')  # 去除标签

    data_list = []
    for i in range(0, 10):  # 总共？？页
        url = base_url1 + str(i * 20) + base_url2   # 更新url,每页有20篇文章
        time.sleep(5)
        html = url_request(url)
        soup = BeautifulSoup(html, 'lxml')

        # 找到每一个影评项
        for item in soup.find_all('div', class_='comment'):
            data = []
            item = str(item)  # 转换成字符串
            # print item

            author = re.findall(pattern_author, item)[0]
            # print author
            data.append(author)  # 添加作者

            star = re.findall(pattern_star, item)
            # print star
            data.append(star)  # 添加推荐等级

            use = re.findall(pattern_use, item)
            # 有用数可能为0，就找不到
            if len(use) != 0:
                use = use[0]
            else:
                use = 0
            # print use
            data.append(use)  # 添加有用数
            data_list.append(data)

            # 从当前网页中爬取相应数据
            review_content = re.findall(pattern_content, item)[0]
            # print review_content
            review_content = re.sub(remove, '', str(review_content))  # 去掉标签
            data.append(review_content)  # 添加评论正文

    return data_list


# 将相关数据写入excel中
def save_data(data_list, save_path):

    num = len(data_list)
    print num
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet('豆瓣影评数据', cell_overwrite_ok=True)
    col = ('作者', '推荐级', '有用数', '影评')
    for i in range(0, 4):
        sheet.write(0, i, col[i])  # 列名
    for i in range(0, num):  # 总共1075条影评
        data = data_list[i]
        for j in range(0, 4):
            sheet.write(i+1, j, data[j])  # 数据
    book.save(save_path)  # 保存


def main():
    base_url1 = 'https://movie.douban.com/subject/25815034/comments?start='
    base_url2 = '&limit=20&sort=new_score'
    data_list = get_data(base_url1, base_url2)
    print len(data_list)
    save_path = u'豆瓣影评数据.xls'
    save_data(data_list, save_path)

if __name__ == "__main__":
    main()