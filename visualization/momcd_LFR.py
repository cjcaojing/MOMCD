import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
# import seaborn as sns
from matplotlib.pyplot import MultipleLocator

plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False   #这两行需要手动设置


def excel_one_line_to_list(j):
    df_news = pd.read_excel(r'lfr_all.xlsx',header = None)
    list=[]
    for i in df_news[j]:
        list.append(i)
    return list

mwlp=excel_one_line_to_list(0)
edmot=excel_one_line_to_list(1)
linlog=excel_one_line_to_list(2)
Motif_SC=excel_one_line_to_list(3)
gemfp=excel_one_line_to_list(4)
etmcd=excel_one_line_to_list(5)
momcd=excel_one_line_to_list(6)


# labels = ['MSOSCD', 'MWLP', 'EdMot-Louvain', 'Motif-LinLog', 'Motif-SC', 'Motif-DECD']
X = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# plt.style.reload_library()  # 重新加载样式
# ['science','ieee'(4色),'nature','scatter'(7色),'grid','notebook','dark_background','high-vis'(6色),'bright'(7色),'vibrant'(7色),'muted'(10色),'high-contrast'(3色),'light'(9色),'std-colors'(7色),'retro'(6色),'']
with plt.style.context(['science', 'no-latex', 'ieee']):  # 使用指定绘画风格 'no-latex’:默认使用latex，则需要电脑安装下载LaTeX
    plt.rcParams.update({
        "font.family": "Times New Roman",  # 字体系列 默认的衬线字体
        "font.serif": ["Times"],  # 衬线字体，Times为Times New Roman
        "font.size": 8})  # 字体大小
    fig, ax = plt.subplots()
    # ax.plot(X, FMMEM, marker='D', markersize=3,linewidth = 1.5, color='red', label='FMMEM')
    # ax.plot(X, LEIDEN, marker='s', markersize=3, color='black',label='Leiden')
    # ax.plot(X, LOUVAIN, marker='o', markersize=3, color='blue', label='Louvain')
    
    ax.plot(X, momcd, marker='D', markersize=3,linewidth = 1.75, color='red', label='MOMCD')
    ax.plot(X,etmcd, marker='*', markersize=3, color='cyan',linestyle='--', label='TMDCD')

    ax.plot(X, edmot, marker='s', markersize=3, color='blue',linestyle=':',label='EdMot')
    ax.plot(X, linlog, marker='1', markersize=5,color='green', label='Linlog-motif')


    ax.plot(X, Motif_SC, marker='<', markersize=3, color='brown', linestyle='-.', label='ME+k-means')
    ax.plot(X, gemfp, marker='v', markersize=3, color='orange', linestyle=':', label='GEMFP')
    ax.plot(X,mwlp, marker='o', markersize=3, color='black', label='MWLP')



    ax.legend(loc='lower left')
    # ax.legend()  # 标识信息 title='Title',loc='upper right'
    ax.set(xlabel='Mixing Parameter (μ)')  # x轴的标题
    ax.set(ylabel='Normalized Mutual Information (NMI)')  # y轴的标题


    # ax.autoscale(tight=True) # 自动缩放：是否紧密，最后一个刻度为图边缘
    # ax.set_xlim(0,1) # x轴的刻度范围
    plt.ylim(-0.04, 1.04)
    x_major_locator = MultipleLocator(0.1)
    y_major_locator = MultipleLocator(0.1)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    fig.savefig(f'momcd_all_nmi.pdf', dpi=1600)  # 输出
