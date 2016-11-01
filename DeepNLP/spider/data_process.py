# coding=utf-8
import xlrd

'''读取评论新闻数据，并按照评分将其分成正向和负向两个文本'''


def read_data(file_path, sheet_name):
    work_book = xlrd.open_workbook(file_path)
    work_sheet = work_book.sheet_by_name(sheet_name)
    data = []
    pos_res = []
    neg_res = []
    # 遍历sheet1中所有行row
    num_rows = work_sheet.nrows

    for curr_row in range(num_rows):

        if curr_row == 0:
            continue
        row = work_sheet.row_values(curr_row)
        # 星级大于30的为正向情感评论
        if row[1] >= '30':
            pos_res.append(row[3].replace(' ', '').replace('\n', '').replace('\t', '').replace('\r', ''))
        # 星级小于30的为负评论
        if row[1] < '30':
            neg_res.append(row[3].replace(' ', '').replace('\n', '').replace('\t', '').replace('\r', ''))
        # data.append(pos_res, neg_res)
    # print('row%s is %s' % (curr_row, row))
    return pos_res, neg_res

    # # 遍历sheet1中所有列col
    # num_cols = work_sheet.ncols
    # for curr_col in range(num_cols):
    #     col = work_sheet.col_values(curr_col)
    # print('col%s is %s' % (curr_col, col))
    #
    # # 遍历sheet1中所有单元格cell
    # for rown in range(num_rows):
    #     for coln in range(num_cols):
    #         cell = work_sheet.cell_value(rown, coln)
    # print cell


def read_data_test():
    file_path = '/Users/li/workshop/MyRepository/DeepNaturalLanguageProcessing/DeepNLP/spider/豆瓣影评数据.xls'
    tmp = read_data(file_path, u'豆瓣影评数据')
    # pos_res = []
    # neg_res = []
    # for i in tmp:
    #     if i[1] >= '30':
    #         pos_res.append(i[3])
    #
    # for i in tmp:
    #     if i[1] < '30':
    #         neg_res.append(i[3])
    #
    # print len(pos_res), len(neg_res), len(tmp)
    for i in tmp[0]:
        print i

    print('\n')

    for i in tmp[0]:
        print i

if __name__ == "__main__":
    read_data_test()

