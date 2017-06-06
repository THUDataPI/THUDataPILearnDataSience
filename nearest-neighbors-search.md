
# 最近邻搜索 (nearest neighbor search)

## <span id = "目录">目录</span>

+ 最近邻搜索基础
    1. [最近邻搜索简介和算法](#一)
    2. [数据表示和距离衡量标准的重要性](#二)  
+ 快速最近邻搜索  
    4. [KD-树：中低维和近似最近邻](#KD-tree)  
    5. [LSH：高维上的近似最近邻](#LSH)  
+ 在Wikipedia上的数值实验  
    3. [Programming Assignment1：不同的数据表示和距离标准](#Assignment1)  
    6. [Programming Assignment2：LSH](#Assignment2)  

<h2 id = "一">一、最近邻搜索简介和算法</h2>

### 1. 问题背景

我们通过解决检索兴趣相关文章的问题来讨论最近邻搜索。首先我们来描述一下文献检索这个问题，假设你正在阅读一篇文章，比如一篇关于体育报道的新闻。   

** 目标：** 找到和当前阅读的文章最相似的文章  
** 两个问题：**

- 怎样定义相似性 (similarity)
- 怎么从一大堆文献中搜索相似文章

### 2. 最近邻搜索 (NN Search)

** 1-最近邻搜索 (1-NN search) **

假设我们已经对所有要被检索的文章定义了一个空间，空间中每篇文章的距离用相似性来描述。然后我们要找到空间中和当前阅读文章(query article)最相似的文章(1-NN)，或者最相似的文章集合。

![文章距离空间](image/article_space.JPG "文章距离空间")

***
** 1-最近邻算法 (1-NN algorithm) **
***
** 输入：** 当前阅读的文章：$X_q$
            文集：$X_1,X_2,\cdots,X_N$   
** 输出：** 最相似的文章：$X^{NN}$  
其中，$X^{NN}$ = arg$\min\limits_{i}$ distance$(X_i,X_q)$  
初始化 ** Dist2NN = $\infty, X^{NN}$ = $\varnothing$ **  
For $i=1,2,\cdots,N$  
　　计算：$\delta = distance(X_i,X_q)$  
    if $\delta<$**Dist2NN**  
    　　令 $X^{NN}=X_i$  
          　　令** Dist2NN**=$\delta$  
Return 最相似文章：$X^{NN}$
***

***
** K-最近邻算法 (K-NN algorithm) **
***  
** 输入：** 当前阅读的文章：$X_q$
             文集：$X_1,X_2,\cdots,X_N$   
** 输出：** 最相似的文章列表(list)：$X^{NN}$  
其中，$X^{NN}$ = $\{X^{NN_1},\cdots,X^{NN_k}\}$，对所有不在$X^{NN}$中的$X_i$，都有$distance(X_i,X_q)\ge\max\limits_{X^{NN_j},j=1,\cdots,k}distance(X^{NN_j},X_q)$  
初始化 ** Dist2KNN ** = sort$(\delta_1,\cdots,\delta_k)$，$X^{NN}=sort(X_1,\cdots,X_k)$     
For $i=k+1,\cdots,N$  
　　计算：$\delta = distance(X_i,X_q)$  
    if $\delta<$**Dist2KNN**[K]  
    　　找到**j**使得**Dist2KNN**[j]>$\delta>$**Dist2KNN**[j-1]  
    　　删除最远的元素并且移动列表：  
        　　　　$X^{NN}[j+1:K]=X^{NN}[j:K-1]$  
            　　**Dist2KNN**[j+1:K]=Dist2KNN[j:K-1]  
        　　   令 **Dist2KNN**[j]=$\delta$，$X^{NN}$[j]=$X_i$    
Return K篇最相似的文章：$X^{NN}$  
***

从上述算法中可以发现，实现**NN Search**的过程比较直观，但是有**两个很关键的因素：**  

- 怎样表示每篇文章，以便计算
- 怎么衡量文章之间的距离

<h2 id = "二">二、数据表示和距离衡量标准的重要性</h2>

### 1. 文章表示

** 简单词数统计(word counts)表示： **

- 忽略词的语序
- 统计每个词出现的次数  

**例：**"Carlos calls the sport futbol. Emily calls the sport soccer."    
$X_q=$ {'Carlos':1, 'the':2, 'Emily':1, 'soccer':1, 'calls':2, 'sport':2, 'futbol':1}

但是这种简单地统计文章中词汇出现的次数的表示方法，有一个缺陷。我们将文章中出现的词主要分为两类： 
+ 常用词：比如，'the', 'player', 'field'这些在所有要搜索文章中都经常出现。
+ 重要稀有词：比如，'futbol', 'Messi'，这些词并不是在所有文章中都频繁出现，但是同时在当前阅读的这篇文章中多次出现。  

显然，常用词对判断两篇文章的相似性没有什么意义，而它们往往又大量出现，所以改进的表示方法中应该增加重要稀有词的权重。  

** TF-IDF 文章表示 (Term Frequency Inverse Document Frequency)：**
强调重要稀有词：
+ 在当前文章中经常出现：
Term Frequency(tf) = 词数
+ 在整个文集中出现较少：
Inverse doc freq.(idf) = $log\frac{文章数}{1+用了这个词的文章数}$   
这一步即为降低常用词的比重，增加重要词汇的权重  

TF-IDF = tf * idf

### 2.欧几里得距离 (Euclidean)和带权的欧几里得（Scaled-Euclidean) 

当数据有多个特征时，或者说表示数据的向量是多维的，我们可能会决定每个特征的重要性，即设定不同的权重。
+ 比如这里，比较两篇文章的相似度时，假设文章的标题和摘要比文章的主体部分更能反映文章内容。
+ 也有可能，特征A变化大，特征B变化小，而实际上两个特征的变化同样重要，就需要对B赋予更大的权重。  
  在这种情况下，一种做法是将权重表示成特征散度的函数，比如，
  第j个特征的权重 = $\frac{1}{\max\limits_{i}(X_i[j])-\min\limits_{i}(X_i[j])}$
  
** Scaled-Euclidean: ** $distance(X_i,X_q)=\sqrt{a_1(X_i[1]-X_q[1])^2+\cdots+a_d(X_i[d]-X_q[d])^2}$，其中，$a_j,j=1,\cdots,d$即为特征的权重。  
特别地，$a_j$取0或1时，就是特征选择的效果。

如何设置权重，或者特征选择/[特征工程](https://www.zhihu.com/question/29316149 "特征工程-知乎")（Feature Selection/Feature Engineering)非常重要但同时也很困难，对不同的问题要具体分析。  

### 3. Cosine Similarity

除了欧几里得距离外，还有一种计算简便的内积距离测量方式：$distance(X_q,X_i)=X_i^{T} X_q$。此时，计算的是两篇文章重叠的词汇。  

**例：** $X_q=[1,0,0,0,5,3,0,0,1,0,0,0,0]$，$X_i=[3,0,0,0,2,0,0,1,0,1,0,0,0]$，相似性=13。

但是，从上面的例子可以看出，如果这两篇文章内容不变地复制成两倍长度的文章，相似性=52，也就是相似性增加，这明显是不合理的。所以考虑标准化的处理方式。

**cosine similarity =** $(\frac{X_i}{||X_i||})^T(\frac{X_q}{||X_q||})=\cos(\theta)$  
不过，cosine similarity并不是一种标准距离，不满足三角不等式，但是在计算稀疏向量（有很多个0）是非常快。
![cosine_similarity](image/cosine_similarity.JPG "cosine_similarity")

### 4. 是否标准化及其他距离

不过**cosine similarity**也有问题，判断两篇文章是否相似时，不是总想和长度无关，也可能让实际不想似的文章变得相似。那么到底要不要标准化呢？实际上，一般会给文章长度设定范围，最短最长是多少。

其他，还有诸如Mahalanobis, rank-based, correlation-based, Manhattan, Jaccard, Hamming等等各种距离度量方法，有兴趣的可以自行查看。  
当然，也可以组合使用这些度量方法，对特征的不同子集应用不同的测量，然后根据具体问题设置不同的权重。

<h2 id = "Assignment1">Programming Assignment1</h2>

[返回目录](#目录)

1. [准备：工具包、加载数据](#准备)
2. [初步实现最近邻搜索：word count](#word count)
3. [TF-IDF改进](#TF-IDF)
4. [不同距离度量的影响](#distance)

通过上述的各种讨论，当我们在大量文章集合中检索时，为了找到相似文章，通常要：

+ 决定相似性的衡量标准
+ 找到最相似的文章

接下来，我们在Wikipedia上实践算法找到和描述总统奥巴马的文章最相关的文章，直观感受一下：

+ 不同的相似性定义标准，怎样找到相似文章
+ 用简单词数统计和TF-IDF来表示文章的不同
+ 不同的距离度量方式，有怎样不同的效果

<h3 id = "准备">相关工具包</h3>

下面的代码在python和[graphlab create](https://turi.com/learn/ "安装graphlab")工具包的环境下运行。

首先，检查一下运行环境和工具包都能正常导入。


```python
import graphlab
import matplotlib.pyplot as plt
import numpy as np
#show plot in jupyter notebook
%matplotlib inline 

'''Check GraphLab Create version'''
from distutils.version import StrictVersion
assert (StrictVersion(graphlab.version) >= StrictVersion('1.8.5')), 'GraphLab Create must be version 1.8.5 or later.'
```

### 加载Wikipedia数据集

在数据集中每个数据由文章的链接，文章描述的人物名字和文章内容（全都转换成小写）组成。


```python
wiki = graphlab.SFrame('people_wiki.gl')
```

    This non-commercial license of GraphLab Create for academic use is assigned to 2903199856@qq.com and will expire on May 31, 2018.
    

    [INFO] graphlab.cython.cy_server: GraphLab Create v2.1 started. Logging: C:\Users\lenovo\AppData\Local\Temp\graphlab_server_1496551835.log.0
    


```python
wiki
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">URI</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">text</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Digby_Morrell&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Digby Morrell</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">digby morrell born 10<br>october 1979 is a former ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Alfred_J._Lewy&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Alfred J. Lewy</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">alfred j lewy aka sandy<br>lewy graduated from ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Harpdog_Brown&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Harpdog Brown</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">harpdog brown is a singer<br>and harmonica player who ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Franz_Rottensteiner&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Franz Rottensteiner</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">franz rottensteiner born<br>in waidmannsfeld lower ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/G-Enka&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">G-Enka</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">henry krvits born 30<br>december 1974 in tallinn ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Sam_Henderson&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Sam Henderson</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">sam henderson born<br>october 18 1969 is an ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Aaron_LaCrate&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Aaron LaCrate</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">aaron lacrate is an<br>american music producer ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Trevor_Ferguson&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Trevor Ferguson</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">trevor ferguson aka john<br>farrow born 11 november ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Grant_Nelson&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Grant Nelson</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">grant nelson born 27<br>april 1971 in london  ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Cathy_Caruth&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Cathy Caruth</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">cathy caruth born 1955 is<br>frank h t rhodes ...</td>
    </tr>
</table>
[59071 rows x 3 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



<h3 id = "word count">提取简单字数统计向量</h3>

用graphlab进行简单计算统计，并作为wiki的一列添加。


```python
wiki['word_count'] = graphlab.text_analytics.count_words(wiki['text'])
```


```python
wiki
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">URI</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">text</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">word_count</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Digby_Morrell&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Digby Morrell</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">digby morrell born 10<br>october 1979 is a former ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'since': 1L, 'carltons':<br>1L, 'being': 1L, '2005': ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Alfred_J._Lewy&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Alfred J. Lewy</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">alfred j lewy aka sandy<br>lewy graduated from ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'precise': 1L, 'thomas':<br>1L, 'closely': 1L, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Harpdog_Brown&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Harpdog Brown</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">harpdog brown is a singer<br>and harmonica player who ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'just': 1L, 'issued':<br>1L, 'mainly': 1L, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Franz_Rottensteiner&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Franz Rottensteiner</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">franz rottensteiner born<br>in waidmannsfeld lower ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'all': 1L,<br>'bauforschung': 1L, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/G-Enka&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">G-Enka</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">henry krvits born 30<br>december 1974 in tallinn ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'legendary': 1L,<br>'gangstergenka': 1L, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Sam_Henderson&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Sam Henderson</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">sam henderson born<br>october 18 1969 is an ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'now': 1L, 'currently':<br>1L, 'less': 1L, 'being': ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Aaron_LaCrate&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Aaron LaCrate</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">aaron lacrate is an<br>american music producer ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'exclusive': 2L,<br>'producer': 1L, 'tribe': ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Trevor_Ferguson&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Trevor Ferguson</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">trevor ferguson aka john<br>farrow born 11 november ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'taxi': 1L, 'salon': 1L,<br>'gangs': 1L, 'being': ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Grant_Nelson&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Grant Nelson</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">grant nelson born 27<br>april 1971 in london  ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'houston': 1L,<br>'frankie': 1L, 'labels': ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Cathy_Caruth&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Cathy Caruth</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">cathy caruth born 1955 is<br>frank h t rhodes ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'phenomenon': 1L,<br>'deborash': 1L, ...</td>
    </tr>
</table>
[59071 rows x 4 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



### 找到相似文章

首先用简单字数统计向量代表文章，通过欧几里得距离计算相似性，来找奥巴马的类似文章。同样用graphlab来实现最近邻搜索算法。


```python
# every data has features: word_count, using euclidean distance to compute similarity
# search method here is brute force.
model = graphlab.nearest_neighbors.create(wiki, label='name', features=['word_count'],
                                          method='brute_force', distance='euclidean')
```


<pre>Starting brute force nearest neighbors model training.</pre>


来看一下最相似的10篇文章：


```python
model.query(wiki[wiki['name']=='Barack Obama'], label='name', k=10)
```


<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.00169288  | 44.002ms     |</pre>



<pre>| Done         |         | 100         | 477.027ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>





<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">query_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">reference_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">distance</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rank</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Joe Biden</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">33.0756708171</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">George W. Bush</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">34.3947670438</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Lawrence Summers</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36.1524549651</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Mitt Romney</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36.1662826401</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Francisco Barrio</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36.3318042492</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Walter Mondale</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36.4005494464</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Wynn Normington Hugh-<br>Jones ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36.4965751818</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Don Bonker</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36.633318168</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Andy Anstett</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36.9594372252</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10</td>
    </tr>
</table>
[10 rows x 4 columns]<br/>
</div>



发现者十个人都是政治家，但是有一半的人除了是政治家以外和Obama就没有什么确切相似的关系了：

+ Francisco Barrio是一个墨西哥政客，是Chihuahua的前州长。
+ Walter Mondale和Don Bonker是1970末的民主党人士。
+ Wynn Normington Hugh-Jones是前英国外交官和自由党官员
+ Andy Anstett是加拿大前政员。

所以，通过简单字数统计来进行最近邻搜索，能返回一些正确信息，但是漏掉了其他一些更重要的细节。

比如，为什么很奇怪地认为Francisco Barrio比其他人和Obama更相关。首先，看一看描述他们二者的文章中出现的最频繁的一些词。


```python
def top_words(name):
    """
    Get a table of the most frequent words in the given person's wikipedia page.
    """
    row = wiki[wiki['name'] == name]
    word_count_table = row[['word_count']].stack('word_count', new_column_name=['word','count'])
    return word_count_table.sort('count', ascending=False) 
```


```python
obama_words = top_words('Barack Obama')
obama_words
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">word</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">count</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">the</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">40</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">in</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">30</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">and</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">21</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">of</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">18</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">to</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">14</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">his</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">11</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">act</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">a</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">he</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
    </tr>
</table>
[273 rows x 2 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>




```python
barrio_words = top_words('Francisco Barrio')
barrio_words
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">word</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">count</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">the</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">of</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">24</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">and</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">18</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">in</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">17</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">he</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">to</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">chihuahua</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">a</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">governor</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">his</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
</table>
[225 rows x 2 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



再来看一下在两篇文章中都经常出现的词。这里用到了**join**操作，相当于两个表通过共有的列进行连接。更多详细类容参考[这篇文章](https://dato.com/products/create/docs/generated/graphlab.SFrame.join.html "SFrame.join")。


```python
combined_words = obama_words.join(barrio_words, on='word')
combined_words
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">word</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">count</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">count.1</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">the</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">40</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">in</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">30</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">17</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">and</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">21</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">18</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">of</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">18</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">24</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">to</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">14</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">his</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">11</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">a</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">he</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">as</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">was</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
    </tr>
</table>
[56 rows x 3 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



为了阅读方便，给combined_word表的两个count列重新命名：


```python
combined_words = combined_words.rename({'count':'Obama', 'count.1':'Barrio'})
combined_words
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">word</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">Obama</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">Barrio</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">the</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">40</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">in</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">30</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">17</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">and</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">21</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">18</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">of</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">18</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">24</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">to</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">14</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">his</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">11</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">a</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">he</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">as</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">was</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
    </tr>
</table>
[56 rows x 3 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



**注意：** **join**操作只是简单地连接相同的列，并没有强制排序，为了看的更清楚，对表进行**sort**操作。


```python
combined_words.sort('Obama', ascending=False)
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">word</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">Obama</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">Barrio</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">the</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">40</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">in</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">30</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">17</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">and</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">21</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">18</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">of</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">18</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">24</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">to</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">14</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">his</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">11</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">a</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">he</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">as</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">was</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
    </tr>
</table>
[56 rows x 3 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



**小问题： **在同时出现在Obama和Barrio文章的单词中，找到在Obama文章中最常出现的5个。那么在数据集，也就是所有文章中，包含了这5个词的有多少篇？

提示：

+ 从上面一块代码中找到两篇文章共有的词，排序找到频数最大的5个。
+ word_count列中每一个元素都是一个字典（dictionary）。对每一个word count值，检查是否有这5个词。定义has_top_words()函数来完成。
    + 将这5个词组成的列表转化成集合：set(common_word)。
    + 从文章的word_count字典中提取关键字列表：keys()。
    + 同样把这些关键字的列表转化成集合。
    + 用[issubset()](https://docs.python.org/2/library/stdtypes.html#set "issubset()")来检查那5个词是否包含在关键字中。
+ 对数据集的每一行，应用has_top_words。
+ 计算有这5个词的文章数目。


```python
common_words = combined_words['word'][:5]  # the largest 5 common words

def has_top_words(word_count_vector):
    # extract the keys of word_count_vector and convert it to a set
    unique_words = word_count_vector.keys() # the words of an article
    # return True if common_words is a subset of unique_words
    # return False otherwise
    return set(common_words).issubset(set(unique_words))  # is that article has common words?

wiki['has_top_words'] = wiki['word_count'].apply(has_top_words)

# use has_top_words column to answer the quiz question
artNums = len(wiki[wiki['has_top_words'] == True])# the sum
```


```python
artNums
```




    56066



随机选两篇文章，来测试一下has_top_words函数。


```python
print 'Output from your function:', has_top_words(wiki[32]['word_count'])
print 'Correct output: True'
print 'Also check the length of unique_words. It should be 167'
```

    Output from your function: True
    Correct output: True
    Also check the length of unique_words. It should be 167
    


```python
print 'Output from your function:', has_top_words(wiki[33]['word_count'])
print 'Correct output: False'
print 'Also check the length of unique_words. It should be 188'
```

    Output from your function: False
    Correct output: False
    Also check the length of unique_words. It should be 188
    

**小问题：**计算介绍Barack Obama, George W. Bush, 和 Joe Biden的文章两两直间的距离，哪个最小？

提示：用graphlab.toolkits.distances.euclidean来计算距离，[参考](https://dato.com/products/create/docs/generated/graphlab.toolkits.distances.euclidean.html "计算距离")。


```python
obama_word_count = wiki[wiki['name']=='Barack Obama']['word_count'][0]
bush_word_count = wiki[wiki['name']=='George W. Bush']['word_count'][0]
biden_word_count = wiki[wiki['name']=='Joe Biden']['word_count'][0]
obama2bush = graphlab.toolkits.distances.euclidean(obama_word_count ,bush_word_count)
obama2biden = graphlab.toolkits.distances.euclidean(obama_word_count ,biden_word_count)
bush2biden = graphlab.toolkits.distances.euclidean(bush_word_count ,biden_word_count)
print 'obama2bush: ', obama2bush
print 'obama2biden: ', obama2biden
print 'bush2biden: ', bush2biden
```

    obama2bush:  34.3947670438
    obama2biden:  33.0756708171
    bush2biden:  32.7566787083
    

**小问题：**在介绍Barack Obama和 George W. Bush的文章都有的词汇里，找出在Obama的文章里最常见的10个词。


```python
bush_words = top_words('George W. Bush')
combined_words = obama_words.join(bush_words, on='word') 
combined_words = combined_words.rename({'count':'Obama', 'count.1':'W.Bush'})
combined_words.sort('Obama', ascending=False)
combined_words['word'][:10]
```




    dtype: str
    Rows: 10
    ['the', 'in', 'and', 'of', 'to', 'he', 'his', 'a', 'president', 'as']



**注意：**即使通用词占了很大的比例，在特别的政治词汇中常出现的词，比如‘president'还是体现出来了。这也就是为什么在反馈和Obama相关的文章中列出的都是政治家而不是音乐家之类的。接下来，我们用TF-IDF来表示文章，将会更强调重要的稀有词的作用。

<h3 id = "TF-IDF">用 **TF-IDF**来改进</h3>

通过上面的展示，发现Obama和Barrio的相似性大多来源于通用词的重复，下面用TF-IDF来表示文章，以强调文章中的重要稀有词，同样搜索和Obama最相关的10篇文章。


```python
wiki['tf_idf'] = graphlab.text_analytics.tf_idf(wiki['word_count'])
```


```python
model_tf_idf = graphlab.nearest_neighbors.create(wiki, label='name', features=['tf_idf'],
                                                 method='brute_force', distance='euclidean')
```


<pre>Starting brute force nearest neighbors model training.</pre>



```python
model_tf_idf.query(wiki[wiki['name'] == 'Barack Obama'], label='name', k=10)
```


<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.00169288  | 3.86s        |</pre>



<pre>| Done         |         | 100         | 4.50s        |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>





<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">query_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">reference_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">distance</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rank</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Phil Schiliro</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">106.861013691</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Jeff Sessions</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">108.871674216</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Jesse Lee (politician)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">109.045697909</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Samantha Power</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">109.108106165</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Bob Menendez</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">109.781867105</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Eric Stern (politician)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">109.95778808</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">James A. Guest</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">110.413888718</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Roland Grossenbacher</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">110.4706087</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Tulsi Gabbard</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">110.696997999</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10</td>
    </tr>
</table>
[10 rows x 4 columns]<br/>
</div>



让我们先判断一下这里列出的文章是否有意义。

+ 除了Roland Grossenbacher，其他的8位都是和Obama同时期的政客。
+ Phil Schiliro, Jesse Lee, Samantha Power, 和 Eric Stern都为Obama工作。

很明显，这个结果比之前用简单字数统计得到的要合理的多。可以详细看一下Obama和Schiliro的文章用词。


```python
def top_words_tf_idf(name):
    row = wiki[wiki['name'] == name]
    word_count_table = row[['tf_idf']].stack('tf_idf', new_column_name=['word','weight'])
    return word_count_table.sort('weight', ascending=False)
```


```python
obama_tf_idf = top_words_tf_idf('Barack Obama')
obama_tf_idf
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">word</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">weight</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">43.2956530721</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">act</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">27.678222623</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">iraq</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">17.747378588</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">control</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">14.8870608452</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">law</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">14.7229357618</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">ordered</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">14.5333739509</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">military</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">13.1159327785</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">involvement</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">12.7843852412</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">response</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">12.7843852412</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">democratic</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">12.4106886973</td>
    </tr>
</table>
[273 rows x 2 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>




```python
schiliro_tf_idf = top_words_tf_idf('Phil Schiliro')
schiliro_tf_idf
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">word</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">weight</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">schiliro</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">21.9729907785</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">staff</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">15.8564416352</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">congressional</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">13.5470876563</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">daschleschiliro</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10.9864953892</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9.62125623824</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">waxman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9.04058524017</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">president</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9.03358661416</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2014from</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8.68391029623</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">law</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7.36146788088</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">consultant</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6.91310403725</td>
    </tr>
</table>
[119 rows x 2 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



同样，用**join**来比较两者的共同词汇，并且按Obama的TF-IDF值来排序。


```python
combined_tf_idf = obama_tf_idf.join(schiliro_tf_idf, on='word') 
combined_tf_idf = combined_tf_idf.rename({'weight':'Obama', 'weight.1':'Schiliro'})
combined_tf_idf.sort('Obama', ascending=False) 
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">word</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">Obama</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">Schiliro</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">43.2956530721</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9.62125623824</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">law</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">14.7229357618</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7.36146788088</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">democratic</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">12.4106886973</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6.20534434867</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">senate</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10.1642881797</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.3880960599</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">presidential</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7.3869554189</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.69347770945</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">president</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7.22686929133</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9.03358661416</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">policy</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6.09538628214</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.04769314107</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">states</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.47320098963</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.82440032988</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">office</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.24817282322</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.62408641161</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2011</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.10704127031</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.40469418021</td>
    </tr>
</table>
[47 rows x 3 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



**小问题：**总共有多少篇文章含有上面那些词里最大的5个呢？


```python
common_words = combined_tf_idf['word'][:5]  

def has_top_words(word_count_vector):
    # extract the keys of word_count_vector and convert it to a set
    unique_words = word_count_vector.keys()   
    # return True if common_words is a subset of unique_words
    # return False otherwise
    return set(common_words).issubset(set(unique_words))  

wiki['has_top_words'] = wiki['word_count'].apply(has_top_words)

# use has_top_words column to answer the quiz question
len(wiki[wiki['has_top_words']==True]) 
```




    14



注意到，用TF-IDF来表示文章，计算距离时排除了通用词的噪声干扰，所以范围缩小了很多。

<h3 id = "distance">选择距离度量方法</h3>

观察返回的相关文章列表，发现Joe Biden，Obama的在两届总统选举中的竞选搭档，竟然不在tf-idf返回的结果中。我们来看一下这是为什么。首先，计算一下这二者的TF-IDF之间的距离。


```python
obama_tf_count = wiki[wiki['name']=='Barack Obama']['tf_idf'][0]
biden_tf_count = wiki[wiki['name']=='Joe Biden']['tf_idf'][0]
obama2biden = graphlab.toolkits.distances.euclidean(obama_tf_count, biden_tf_count)
obama2biden
```




    123.29745600964296



确实，这个距离比返回的列表中的距离要大，我们重新看一下这个列表。


```python
model_tf_idf.query(wiki[wiki['name'] == 'Barack Obama'], label='name', k=10)
```


<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.00169288  | 17.001ms     |</pre>



<pre>| Done         |         | 100         | 485.027ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>





<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">query_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">reference_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">distance</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rank</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Phil Schiliro</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">106.861013691</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Jeff Sessions</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">108.871674216</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Jesse Lee (politician)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">109.045697909</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Samantha Power</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">109.108106165</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Bob Menendez</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">109.781867105</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Eric Stern (politician)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">109.95778808</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">James A. Guest</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">110.413888718</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Roland Grossenbacher</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">110.4706087</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Tulsi Gabbard</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">110.696997999</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10</td>
    </tr>
</table>
[10 rows x 4 columns]<br/>
</div>



那么，是不是关于Biden的文章与Obama的之间的差别，比Schiliro和Obama的更大呢？更进一步分析，发现我们的搜索结果返回了更多短文。但是说短文比长文更相关，明显是没有道理的。先计算一下数据集中所有文章的长度和与Obama最相关的100篇文章的长度。


```python
def compute_length(row):
    return len(row['text'].split(' '))

wiki['length'] = wiki.apply(compute_length) 
```


```python
# find the nearest 100 articles
nearest_neighbors_euclidean = model_tf_idf.query(wiki[wiki['name'] == 'Barack Obama'], label='name', k=100)
# the length of these 100 articles
nearest_neighbors_euclidean = nearest_neighbors_euclidean.join(wiki[['name', 'length']], on={'reference_label':'name'})
```


<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.00169288  | 18.001ms     |</pre>



<pre>| Done         |         | 100         | 593.034ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



```python
nearest_neighbors_euclidean.sort('rank')
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">query_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">reference_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">distance</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rank</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">length</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">540</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Phil Schiliro</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">106.861013691</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">208</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Jeff Sessions</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">108.871674216</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">230</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Jesse Lee (politician)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">109.045697909</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">216</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Samantha Power</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">109.108106165</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">310</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Bob Menendez</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">109.781867105</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">220</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Eric Stern (politician)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">109.95778808</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">255</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">James A. Guest</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">110.413888718</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">215</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Roland Grossenbacher</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">110.4706087</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">201</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Tulsi Gabbard</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">110.696997999</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">228</td>
    </tr>
</table>
[100 rows x 5 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



为了清楚地比较这些相似文章和其他文章的长度，画一个柱状图。


```python
plt.figure(figsize=(10.5,4.5))
plt.hist(wiki['length'], 50, color='k', edgecolor='None', histtype='stepfilled', normed=True,
         label='Entire Wikipedia', zorder=3, alpha=0.8)
plt.hist(nearest_neighbors_euclidean['length'], 50, color='r', edgecolor='None', histtype='stepfilled', normed=True,
         label='100 NNs of Obama (Euclidean)', zorder=10, alpha=0.8)
plt.axvline(x=wiki['length'][wiki['name'] == 'Barack Obama'][0], color='k', linestyle='--', linewidth=4,
           label='Length of Barack Obama', zorder=2)
plt.axvline(x=wiki['length'][wiki['name'] == 'Joe Biden'][0], color='g', linestyle='--', linewidth=4,
           label='Length of Joe Biden', zorder=1)
plt.axis([0, 1000, 0, 0.04])

plt.legend(loc='best', prop={'size':15})
plt.title('Distribution of document length')
plt.xlabel('# of words')
plt.ylabel('Percentage')
plt.rcParams.update({'font.size':16})
plt.tight_layout()
```


![png](output/output_90_0.png)


和数据集中其他文章相比，Obama的相似文章大都相对地短很多，不超过300词。但是Wikipedia中的很多文章都超过了300个单词，而且，Obama和Biden的文章都超过了300，所以认为短文章更相关显然是没有道理的。

**注意：**出于计算时间的考虑，这里的文章其实都只是摘要而非全文，所以比真实的文章短得多。

为了消除这种对短文的偏向，考虑**cosine distances**：$$d(x,y) = 1-\frac{x^Ty}{||x||||y||}$$   
下面训练一个新的最近邻模型，用cosine distance来计算和Obama最相近的100篇文章。


```python
model2_tf_idf = graphlab.nearest_neighbors.create(wiki, label='name', features=['tf_idf'],
                                                  method='brute_force', distance='cosine')
```


<pre>Starting brute force nearest neighbors model training.</pre>



```python
nearest_neighbors_cosine = model2_tf_idf.query(wiki[wiki['name'] == 'Barack Obama'], label='name', k=100)
nearest_neighbors_cosine = nearest_neighbors_cosine.join(wiki[['name', 'length']], on={'reference_label':'name'})
```


<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.00169288  | 690.039ms    |</pre>



<pre>| Done         |         | 100         | 1.08s        |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



```python
nearest_neighbors_cosine.sort('rank')
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">query_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">reference_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">distance</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rank</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">length</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">540</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Joe Biden</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.703138676734</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">414</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Samantha Power</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.742981902328</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">310</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Hillary Rodham Clinton</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.758358397887</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">580</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Eric Stern (politician)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.770561227601</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">255</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Robert Gibbs</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.784677504751</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">257</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Eric Holder</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.788039072943</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">232</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Jesse Lee (politician)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.790926415366</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">216</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Henry Waxman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.798322602893</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">279</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Joe the Plumber</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.799466360042</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">217</td>
    </tr>
</table>
[100 rows x 5 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



看一看上面的这个结果，Joe Biden是最相关的，同时Hillary也上榜了，看起来相当不错。

画个图看一下：


```python
plt.figure(figsize=(10.5,4.5))
plt.figure(figsize=(10.5,4.5))
plt.hist(wiki['length'], 50, color='k', edgecolor='None', histtype='stepfilled', normed=True,
         label='Entire Wikipedia', zorder=3, alpha=0.8)
plt.hist(nearest_neighbors_euclidean['length'], 50, color='r', edgecolor='None', histtype='stepfilled', normed=True,
         label='100 NNs of Obama (Euclidean)', zorder=10, alpha=0.8)
plt.hist(nearest_neighbors_cosine['length'], 50, color='b', edgecolor='None', histtype='stepfilled', normed=True,
         label='100 NNs of Obama (cosine)', zorder=11, alpha=0.8)
plt.axvline(x=wiki['length'][wiki['name'] == 'Barack Obama'][0], color='k', linestyle='--', linewidth=4,
           label='Length of Barack Obama', zorder=2)
plt.axvline(x=wiki['length'][wiki['name'] == 'Joe Biden'][0], color='g', linestyle='--', linewidth=4,
           label='Length of Joe Biden', zorder=1)
plt.axis([0, 1000, 0, 0.04])
plt.legend(loc='best', prop={'size':15})
plt.title('Distribution of document length')
plt.xlabel('# of words')
plt.ylabel('Percentage')
plt.rcParams.update({'font.size': 16})
plt.tight_layout()
```


    <matplotlib.figure.Figure at 0x18a91a58>



![png](output/output_98_1.png)


确实，cosine distance反馈的100篇文章基本上是在整个数据集各种长度的文章上的一个采样，而不是像欧几里得距离那样集中在短文章中。

**启示：**在决定特征和距离度量方法时，检查你得到的结果是否满足了你的要求，是否合理。

### cosine distance的问题：tweets V.S. 长文章

还没有完，前面提到cosine distance忽略了所有的文章长度，而这并不是在所有问题中都适用的，比如下面这个特地构造出来的一条tweet。

```
+--------------------------------------------------------+
|                                             +--------+ |
|  One that shall not be named                | Follow | |
|  @username                                  +--------+ |
|                                                        |
|  Democratic governments control law in response to     |
|  popular act.                                          |
|                                                        |
|  8:05 AM - 16 May 2016                                 |
|                                                        |
|  Reply   Retweet (1,332)   Like (300)                  |
|                                                        |
+--------------------------------------------------------+
```

这条tweet和介绍Obama的Wikipedia文章有多相似呢？先把tweet转化成数据集中数据的形式，计算tf-idf。


```python
sf = graphlab.SFrame({'text': ['democratic governments control law in response to popular act']})
sf['word_count'] = graphlab.text_analytics.count_words(sf['text'])

encoder = graphlab.feature_engineering.TFIDF(features=['word_count'], output_column_prefix='tf_idf')
encoder.fit(wiki)
sf = encoder.transform(sf)
sf
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">text</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">word_count</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">tf_idf.word_count</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">democratic governments<br>control law in response ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'control': 1L,<br>'democratic': 1L, 'act': ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'control':<br>3.721765211295327, ...</td>
    </tr>
</table>
[1 rows x 3 columns]<br/>
</div>



下面是Obama和tweet的TF-IDF向量：


```python
tweet_tf_idf = sf[0]['tf_idf.word_count']
tweet_tf_idf
```




    {'act': 3.4597778278724887,
     'control': 3.721765211295327,
     'democratic': 3.1026721743330414,
     'governments': 4.167571323949673,
     'in': 0.0009654063501214492,
     'law': 2.4538226269605703,
     'popular': 2.764478952022998,
     'response': 4.261461747058352,
     'to': 0.04694493768179923}




```python
obama = wiki[wiki['name'] == 'Barack Obama']
obama
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">URI</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">text</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">word_count</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">has_top_words</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Barack_Obama&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">barack hussein obama ii<br>brk husen bm born august ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'operations': 1L,<br>'represent': 1L, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
</table>
<table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">tf_idf</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">length</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'operations':<br>3.811771079388818, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">540</td>
    </tr>
</table>
[? rows x 7 columns]<br/>Note: Only the head of the SFrame is printed. This SFrame is lazily evaluated.<br/>You can use sf.materialize() to force materialization.
</div>



计算二者之间的cosine distance：


```python
obama_tf_idf = obama[0]['tf_idf']
graphlab.toolkits.distances.cosine(obama_tf_idf, tweet_tf_idf)
```




    0.7059183777794327



和那些与Obama最相关的10篇文章的距离相比较：


```python
model2_tf_idf.query(obama, label='name', k=10)
```


<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.00169288  | 1.08s        |</pre>



<pre>| 0            | 46349   | 78.4632     | 1.48s        |</pre>



<pre>| 0            | 55570   | 94.0732     | 2.67s        |</pre>



<pre>| Done         |         | 100         | 3.08s        |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>





<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">query_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">reference_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">distance</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rank</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Joe Biden</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.703138676734</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Samantha Power</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.742981902328</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Hillary Rodham Clinton</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.758358397887</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Eric Stern (politician)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.770561227601</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Robert Gibbs</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.784677504751</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Eric Holder</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.788039072943</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Jesse Lee (politician)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.790926415366</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Henry Waxman</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.798322602893</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Joe the Plumber</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.799466360042</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10</td>
    </tr>
</table>
[10 rows x 4 columns]<br/>
</div>



发现这条tweet和Obama的距离，除了比不过Joe Biden外，都要小。但是这不能说明tweet就和Obama更相关。在这个例子中，完全忽略长度是不合理的。所以实际上，要有最大和最小文章长度。

<h2 id = "KD-tree">三、用KD-树(KD-trees)扩大K-NN搜索规模</h2>

### brute force搜索的时间复杂度

**brute force**搜索，即直接枚举搜索，一个一个按顺序比较，所以时间复杂度是：

+ 1-NN：O(N)
+ K-NN: O(NlogK)，这里维持K篇当前搜索出的文章距离的顺序需要O(logK)的时间。 

所以，当文章集合非常大，而且对固定的文集进行多次检索时，检索时间很长。

### KD-trees

将数据集表示成一种新的数据结构--**KD-trees**，用这种数据结构来组织文章更高效。首先假设2维的情况，只有两个词，构造KD-trees：

1. 对空间进行轴向切割，以划分空间成不同的盒子(区域)(bins)；
    + 对哪一维分割(这里是X,Y轴)：split dimention
    + 按多少来分割：split value
2. 按照分割，决定数据集中的每一个点落在哪个盒子里；
3. 对每一个盒子或者说数据子集，又进行分割，重复1.2.过程。
    + 直到停止，比如每个盒子里只有少量的点

这样，形成一个二叉树，二叉树的每一个叶子节点里是在同一个盒子里的数据集合，每一个中间节点中则储存分割维度(split dimension)、分割值(split value)和包括当前盒子的所有点的最小范围(bounding box)。

![KD-tree构造](image/KD-constructure.JPG "构造KD-树")

KD-树的构造还是很直观的，关键在于：

+ 每一次怎么选取划分维度：
    选范围最大的维度，交替选择维度...
+ 每一次的划分值怎么确定
    选划分维度的中点，选数据中心...
+ 什么时候停止划分
    每个最小盒子中包含的数据点小于给定值m，盒子的大小小于给定值...
    
这些都没有固定的标准，一般都是实际应用中，探索地设置调整。

![不同的划分值](image/heuristic.jpg "划分方式")

### 在KD-树上进行1-最近邻搜索

构造好KD-tree之后，我们遍历二叉树来搜索。对一篇给定的当前阅读文章(query point)：

1. 沿着二叉树向下查找query point所在的叶子节点。
2. 计算在这个叶子节点中所有点到query point的距离，找到目前为止最近的点。
    
    记此时的最近距离为r，但是并不就是最近点了。  
    
3. 回溯，搜索其他分支，更新当前最近距离。
   
   根据节点中记录的bounding box，如果query point的点到bounding box的距离比r大，那么整个节点分支都可以被删除。
   
从第三步，可以看出如果划分合适的话，利用bounding box可以大大减少要搜索的点，从而提高效率。

对应的K-最近邻搜索，几乎是一样的，只是此时要记录最近的K个点和距离。

### KD-tree来搜索的时间复杂度

KD-tree的搜索效率和分割空间有很大的关系，假设最终得到的是一棵基本平衡的二叉树，我们看两种极端的情况：

![极端时间复杂度](image/complexity.JPG "极端情况")

所以，如果数据集中共有N个点，找到1-NN的时间在O(logN)和O(N)之间。而构造KD-树的时间复杂度为O(NlogN)(这里对树的每一层用到了优先队列)。如果只进行一次搜索，可能不必要花时间构造树，但是如果对同一个数据集进行多次，比如N次，每篇文章都来找最近邻，那么和brute force搜索相比，效率大大提高。

### 在大规模数据集上，KD-树的性能

**KD-树**在中低维数据上，搜索效率高。

然而，正如之前讨论过的，KD-树的优势在于能删除多个分支，从而减少检查的点。在高维数据集上，以当前最近距离为半径的超球面和超面体在多个维度上相交的可能性增加，也就是说不能删除的点的数目变多，就达不到提高效率的目的。

通过数值实验，发现随着维数d的增加，搜索时间以指数速度增长：

![关于N和d的搜索时间](image/time.JPG "搜索时间")

### 用KD-树搜索近似的最近邻

到目前为止，我们都假设：最近邻算法找到最近解比找到其他差不多的解要更有意义，用来计算距离的方法完全有效科学，所以我们找到的就是最理想的最近邻。但实际上，可能没有必要找到最精确的那个解，找一个近似的但足够满足要求的相似文章，从而节省许多的时间。

通过KD-树来近似，r表示当前最近距离，用$\frac{r}{\alpha}$作为约束来删除点。即，如果我们找到了距离是r的最近邻，那么不会有更近的点到query point的距离小于$\frac{r}{\alpha}$。这样，我们节省了时间因为删除了更多的点，而精确度差不多,因为计算距离的时候本来就有许多噪声。

<h2 id = "LSH">四、LSH(Locality Sensitive Hashing)求近似最近邻</h2>

经过讨论，KD-树在处理高维数据时效率低，而且构造KD-树比较麻烦；同时为了节省时间可以不要求精确的解，接下来介绍一种可以控制正确率的近似求解方法--**LSH**。

### 用LSH代替KD-tree来表示数据集

像KD-tree一样，也对空间进行分割，不过并不是轴向分割，这里同样先假设2维的情形。

1. 画一条过原点的直线，简单地将所有点分成两部分。
2. 点在直线上方或下方，计算点的值，规定它们分别属于0号或1号两个盒子。这样得到一个哈希表(hash table)。
3. query point在那个盒子，就搜索哪个盒子的点。

![hash table](image/hash_table.JPG "hash table")

### 随机选择分割线

LSH的思路非常简单，但是这条分割线怎么选才合理呢？假设我们随机画过原点的线，如果用cosine similarity来度量点之间的距离，那么分割线将query point和它的最近邻分开的概率，也就是搜索一个盒子后找不到最近邻的概率应该是**$\frac{2\theta}{\pi}$**。这是一个小概率事件。

所以线简单地随机选择就好了，但是还有一个问题，如果数据集很大，画一条线之后，每个盒子里的点依然很多，那计算量还是很大。

### 划分更多的盒子

为了减少检查的点，选择画多条线。此时，每个点的指标用一个二进制向量表示，对这些二进制向量进行编码，得到每个盒子的编号，也就是哈希表中的关键字。

![多个盒子](image/multiple_lines.JPG "多条分割线")

但是，由于分割线变多，最近邻被分到不同的盒子中的概率变大，这样虽然搜索时间减少，但是正确率下降。

### 搜索相邻的盒子

这里，不进行详细的数学证明，但是直观上容易看出，搜索越多的相邻区域，正确率越高，因为query point和最近邻被多条线分开的概率变小。又由于相邻区域在一条分割线的两边，所以二进制向量的一位变化，就代表一个相邻区域，依次类推。最多搜索完所有区域，找到的一定是精确的最近邻。

在实际计算时，可以设定正确率要求和最大计算时间，来停止邻近区域的搜索。

### LSH处理高维数据

现在，我们来处理实际中的高维数据集：

1. 随机选择多个过原点的超平面；
2. 计算每个点的值，来确定对应的二进制指标0/1;
3. 对query point所在的区域进行搜索，然后是相邻区域，再相邻的区域，反复直到达到停止条件。

对于高维情况，可以粗略地估计一下构造哈希表的时间复杂度：计算每个点的值，假设d维数据，i个划分超平面，进行$d\times i$次乘法。同时因为实际一篇文章中只有有限的不同词汇，所以是一个稀疏向量乘法，那么实际耗时会小得多。在进行多次查询时，这个时间是可以忍受的。

<h2 id = "Assignment2">Programming Assignment2</h2>

[返回目录](#目录)

1. [准备：包、载入数据，提取TF-IDF矩阵](#准备)
2. [训练LSH模型](#训练)
3. LSH效果
    + [对区域的观察](#区域)
    + [不同的搜索半径](#半径)
    + [划分区域的随机向量数目变化](#向量)

下面，仍然以Wikipedia数据集为例：

+ 实现**LSH**，求相似最近邻
+ 对不同的文章，比较**LSH**和**brute force**的准确率和用时
+ 调整算法的参数，探讨对准确性的影响

<h3 id = "准备">导入包并检查环境</h3>


```python
import numpy as np
import graphlab
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import pairwise_distances
import time
from copy import copy
import matplotlib.pyplot as plt
%matplotlib inline

'''Check GraphLab Create version'''
from distutils.version import StrictVersion
assert (StrictVersion(graphlab.version) >= StrictVersion('1.8.5')), 'GraphLab Create must be version 1.8.5 or later.'

'''compute norm of a sparse vector
   Thanks to: Jaiyam Sharma'''
def norm(x):
    sum_sq=x.dot(x.T)
    norm=np.sqrt(sum_sq)
    return(norm)
```

### 载入Wikipedia数据集


```python
wiki = graphlab.SFrame('people_wiki.gl/')
```

    This non-commercial license of GraphLab Create for academic use is assigned to 2903199856@qq.com and will expire on May 31, 2018.
    

    [INFO] graphlab.cython.cy_server: GraphLab Create v2.1 started. Logging: C:\Users\lenovo\AppData\Local\Temp\graphlab_server_1496577373.log.0
    

由于要在哈希表中存储对应的文章编号，所以这里赋予每一个数据一个唯一的ID。


```python
wiki = wiki.add_row_number()
wiki
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">id</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">URI</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">text</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Digby_Morrell&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Digby Morrell</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">digby morrell born 10<br>october 1979 is a former ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Alfred_J._Lewy&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Alfred J. Lewy</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">alfred j lewy aka sandy<br>lewy graduated from ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Harpdog_Brown&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Harpdog Brown</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">harpdog brown is a singer<br>and harmonica player who ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Franz_Rottensteiner&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Franz Rottensteiner</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">franz rottensteiner born<br>in waidmannsfeld lower ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/G-Enka&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">G-Enka</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">henry krvits born 30<br>december 1974 in tallinn ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Sam_Henderson&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Sam Henderson</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">sam henderson born<br>october 18 1969 is an ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Aaron_LaCrate&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Aaron LaCrate</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">aaron lacrate is an<br>american music producer ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Trevor_Ferguson&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Trevor Ferguson</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">trevor ferguson aka john<br>farrow born 11 november ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Grant_Nelson&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Grant Nelson</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">grant nelson born 27<br>april 1971 in london  ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Cathy_Caruth&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Cathy Caruth</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">cathy caruth born 1955 is<br>frank h t rhodes ...</td>
    </tr>
</table>
[59071 rows x 4 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



### 提取TF-IDF矩阵


```python
# compute the tf-idf of each articles
wiki['tf_idf'] = graphlab.text_analytics.tf_idf(wiki['text'])
wiki
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">id</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">URI</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">text</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">tf_idf</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Digby_Morrell&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Digby Morrell</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">digby morrell born 10<br>october 1979 is a former ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'since':<br>1.455376717308041, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Alfred_J._Lewy&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Alfred J. Lewy</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">alfred j lewy aka sandy<br>lewy graduated from ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'precise':<br>6.44320060695519, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Harpdog_Brown&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Harpdog Brown</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">harpdog brown is a singer<br>and harmonica player who ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'just':<br>2.7007299687108643, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Franz_Rottensteiner&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Franz Rottensteiner</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">franz rottensteiner born<br>in waidmannsfeld lower ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'all':<br>1.6431112434912472, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/G-Enka&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">G-Enka</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">henry krvits born 30<br>december 1974 in tallinn ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'legendary':<br>4.280856294365192, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Sam_Henderson&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Sam Henderson</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">sam henderson born<br>october 18 1969 is an ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'now': 1.96695239252401,<br>'currently': ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Aaron_LaCrate&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Aaron LaCrate</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">aaron lacrate is an<br>american music producer ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'exclusive':<br>10.455187230695827, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Trevor_Ferguson&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Trevor Ferguson</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">trevor ferguson aka john<br>farrow born 11 november ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'taxi':<br>6.0520214560945025, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Grant_Nelson&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Grant Nelson</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">grant nelson born 27<br>april 1971 in london  ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'houston':<br>3.935505942157149, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Cathy_Caruth&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Cathy Caruth</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">cathy caruth born 1955 is<br>frank h t rhodes ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'phenomenon':<br>5.750053426395245, ...</td>
    </tr>
</table>
[59071 rows x 5 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



TF-IDF矩阵是一个稀疏矩阵，因为每一篇文章只用到一部分单词，这里用python工具包SciPy来存储稀疏矩阵。许多对NumPy中的数组可以进行的操作对SciPy的稀疏矩阵一样有用。

首先，将TF-IDF列转化成SciPy的稀疏矩阵的形式。


```python
def sframe_to_scipy(column):
    """ 
    Convert a dict-typed SArray into a SciPy sparse matrix.
    
    Returns
    -------
        mat : a SciPy sparse matrix where mat[i, j] is the value of word j for document i.
        mapping : a dictionary where mapping[j] is the word whose values are in column j.
    """
    # Create triples of (row_id, feature_id, count).
    x = graphlab.SFrame({'X1':column})
    
    # 1. Add a row number.
    x = x.add_row_number()
    # 2. Stack will transform x to have a row for each unique (row, key) pair.
    x = x.stack('X1', ['feature', 'value'])

    # Map words into integers using a OneHotEncoder feature transformation.
    f = graphlab.feature_engineering.OneHotEncoder(features=['feature'])

    # We first fit the transformer using the above data.
    f.fit(x)

    # The transform method will add a new column that is the transformed version
    # of the 'word' column.
    x = f.transform(x)

    # Get the feature mapping.
    mapping = f['feature_encoding']

    # Get the actual word id.
    x['feature_id'] = x['encoded_features'].dict_keys().apply(lambda x: x[0])

    # Create numpy arrays that contain the data for the sparse matrix.
    i = np.array(x['id'])
    j = np.array(x['feature_id'])
    v = np.array(x['value'])
    width = x['id'].max() + 1
    height = x['feature_id'].max() + 1

    # Create a sparse matrix.
    mat = csr_matrix((v, (i, j)), shape=(width, height))

    return mat, mapping
```

下面的转换可能要花几分钟。


```python
start=time.time()
corpus, mapping = sframe_to_scipy(wiki['tf_idf'])
end=time.time()
print end-start
```

    262.110999823
    

检查一下是否正确转换了，稀疏矩阵中应该包括59071篇文章和547979个词。


```python
assert corpus.shape == (59071, 547979)
print 'Check passed correctly!'
```

    Check passed correctly!
    

<h3 id = "训练">训练一个LSH模型</h3>

LSH通过随机地将数据集划分到不同的区域，高效地完成近邻搜索。这里我们实现LSH的一种常用形式--random binary projection，逼近cosine distance。

第一步，从标准高斯分布中产生随机向量的集合：


```python
def generate_random_vectors(num_vector, dim):
    return np.random.randn(dim, num_vector)
```

我们可以看一个低维的例子，产生3个5维的随机向量。


```python
# Generate 3 random vectors of dimension 5, arranged into a single 5 x 3 matrix.
np.random.seed(0) # set seed=0 for consistent results
generate_random_vectors(num_vector=3, dim=5)
```




    array([[ 1.76405235,  0.40015721,  0.97873798],
           [ 2.2408932 ,  1.86755799, -0.97727788],
           [ 0.95008842, -0.15135721, -0.10321885],
           [ 0.4105985 ,  0.14404357,  1.45427351],
           [ 0.76103773,  0.12167502,  0.44386323]])



对于我们的文章检索，则是和单词数相同维数的随机向量(547979维)。对应每一个随机向量，可以得到二进制向量中的一位。如果生成16个向量，那么每一篇文章的二进制指标是16位的。


```python
# Generate 16 random vectors of dimension 547979
np.random.seed(0)
random_vectors = generate_random_vectors(num_vector=16, dim=547979)
random_vectors.shape
```




    (547979L, 16L)



第二步，划分点到不同的区域。为了提高计算速度，这里我们用矩阵运算而不是循环来进行划分。

比如，要判断第0篇文章在哪个区域。这篇文章的二进制向量指标的第一位，取决于第一个随机向量和它的TF-IDF的点乘的符号。


```python
doc = corpus[0, :] # vector of tf-idf values for document 0
doc.dot(random_vectors[:, 0]) >= 0 # True if positive sign; False if negative sign
```




    array([ True], dtype=bool)



同理第二位就是与第二个随机向量的点乘。

矩阵运算能快速地经行批量的向量点乘运算，得到16位的值，比循环运算要快得多。


```python
doc.dot(random_vectors) >= 0 # should return an array of 16 True/False bits
```




    array([[ True,  True, False, False, False,  True,  True, False,  True,
             True,  True, False, False,  True, False,  True]], dtype=bool)




```python
np.array(doc.dot(random_vectors) >= 0, dtype=int) # display index bits in 0/1's
```




    array([[1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1]])



有相同的二进制指标的文章在同一个区域。还是用矩阵运算来计算所有文章的指标。


```python
corpus[0:2].dot(random_vectors) >= 0 # compute bit indices of first two documents
```




    array([[ True,  True, False, False, False,  True,  True, False,  True,
             True,  True, False, False,  True, False,  True],
           [ True, False, False, False,  True,  True, False,  True,  True,
            False,  True, False,  True, False, False,  True]], dtype=bool)




```python
corpus.dot(random_vectors) >= 0 # compute bit indices of ALL documents
```




    array([[ True,  True, False, ...,  True, False,  True],
           [ True, False, False, ..., False, False,  True],
           [False,  True, False, ...,  True, False,  True],
           ..., 
           [ True,  True, False, ...,  True,  True,  True],
           [False,  True,  True, ...,  True, False,  True],
           [ True, False,  True, ..., False, False,  True]], dtype=bool)



接下来，为了方便，对这些二进制向量进行编码，以得到整数的哈希表的关键字：
```
Bin index                      integer
[0,0,0,0,0,0,0,0,0,0,0,0]   => 0
[0,0,0,0,0,0,0,0,0,0,0,1]   => 1
[0,0,0,0,0,0,0,0,0,0,1,0]   => 2
[0,0,0,0,0,0,0,0,0,0,1,1]   => 3
...
[1,1,1,1,1,1,1,1,1,1,0,0]   => 65532
[1,1,1,1,1,1,1,1,1,1,0,1]   => 65533
[1,1,1,1,1,1,1,1,1,1,1,0]   => 65534
[1,1,1,1,1,1,1,1,1,1,1,1]   => 65535 (= 2^16-1)
```
这种对应编码可以通过计算二进制向量和2的幂次的向量的点乘得到：


```python
doc = corpus[0, :]  # first document
index_bits = (doc.dot(random_vectors) >= 0)
powers_of_two = (1 << np.arange(15, -1, -1))
print index_bits
print powers_of_two
print index_bits.dot(powers_of_two)
```

    [[ True  True False False False  True  True False  True  True  True False
      False  True False  True]]
    [32768 16384  8192  4096  2048  1024   512   256   128    64    32    16
         8     4     2     1]
    [50917]
    

这又是向量点乘，可以通过矩阵批量运算：


```python
index_bits = corpus.dot(random_vectors) >= 0
index_bits.dot(powers_of_two)
```




    array([50917, 36265, 19365, ..., 52983, 27589, 41449])



现在，我们得到了所有文章的二进制向量和整数指标，和哈希表的整数关键字。接下来，就是根据这些指标来构造哈希表。每个区域指标对应一列文章的标号，所以应该得到一个字典。

1. 计算整数指标，即上面的工作。
2. 对每一篇文章：
    + 获得它的整数指标
    + 找到这个整数指标对应的区域包含的文章列表；如果列表不存在，则为这个区域赋予一个空列
    + 将这篇文章的指标id添加到列表末
    


```python
def train_lsh(data, num_vector=16, seed=None):
    
    dim = data.shape[1]
    if seed is not None:
        np.random.seed(seed)
    random_vectors = generate_random_vectors(num_vector, dim)
  
    powers_of_two = 1 << np.arange(num_vector-1, -1, -1)
  
    table = {}
    
    # Partition data points into bins
    bin_index_bits = (data.dot(random_vectors) >= 0)
  
    # Encode bin index bits into integers
    bin_indices = bin_index_bits.dot(powers_of_two)
    
    # Update `table` so that `table[i]` is the list of document ids with bin index equal to i.
    for data_index, bin_index in enumerate(bin_indices):
        if bin_index not in table:
            # If no list yet exists for this bin, assign the bin an empty list.
            table[bin_index] = [] # YOUR CODE HERE
        # Fetch the list of document ids associated with the bin and add the document id to the end.
        table[bin_index].append(data_index) # YOUR CODE HERE

    model = {'data': data,
             'bin_index_bits': bin_index_bits,
             'bin_indices': bin_indices,
             'table': table,
             'random_vectors': random_vectors,
             'num_vector': num_vector}
    
    return model
```

**检查一下**


```python
model = train_lsh(corpus, num_vector=16, seed=143)
table = model['table']
if   0 in table and table[0]   == [39583] and \
   143 in table and table[143] == [19693, 28277, 29776, 30399]:
    print 'Passed!'
else:
    print 'Check your code.'
```

    Passed!
    

下面的实践，如无特别说明，都在这个模型上完成。

<h3 id = "区域">观察区域</h3>


选一些文章，看看它们落在哪个区域。


```python
wiki[wiki['name'] == 'Barack Obama']
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">id</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">URI</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">text</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">tf_idf</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">35817</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Barack_Obama&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">barack hussein obama ii<br>brk husen bm born august ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'operations':<br>3.811771079388818, ...</td>
    </tr>
</table>
[? rows x 5 columns]<br/>Note: Only the head of the SFrame is printed. This SFrame is lazily evaluated.<br/>You can use sf.materialize() to force materialization.
</div>



**小问题：**Barack Obama的文章的整数指标是多少？


```python
obama_bin_index = model['bin_indices'][35817]
print obama_bin_index
```

    50194
    

根据Programmin Assignment1，Joe Biden是Obama的最近邻。


```python
wiki[wiki['name'] == 'Joe Biden']
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">id</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">URI</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">text</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">tf_idf</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">24478</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Joe_Biden&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Joe Biden</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">joseph robinette joe<br>biden jr dosf rbnt badn ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'delaware':<br>11.396456717061318, ...</td>
    </tr>
</table>
[? rows x 5 columns]<br/>Note: Only the head of the SFrame is printed. This SFrame is lazily evaluated.<br/>You can use sf.materialize() to force materialization.
</div>



**小问题：**Obama和Biden的二进制指标，有多少位是相同的？


```python
print model['bin_index_bits'][24478] == model['bin_index_bits'][35817]
```

    [ True False  True  True  True  True  True  True  True  True  True False
      True  True  True  True]
    

14位相同，再和前英国外交官比较一下，发现只有8位相同。


```python
wiki[wiki['name']=='Wynn Normington Hugh-Jones']
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">id</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">URI</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">text</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">tf_idf</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">22745</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Wynn_Normington_H ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Wynn Normington Hugh-<br>Jones ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">sir wynn normington<br>hughjones kb sometimes ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'forced':<br>3.919175540571719, ...</td>
    </tr>
</table>
[? rows x 5 columns]<br/>Note: Only the head of the SFrame is printed. This SFrame is lazily evaluated.<br/>You can use sf.materialize() to force materialization.
</div>




```python
print np.array(model['bin_index_bits'][22745], dtype=int) # list of 0/1's
print model['bin_indices'][22745] # integer format
model['bin_index_bits'][35817] == model['bin_index_bits'][22745]
```

    [0 0 0 1 0 0 1 0 0 0 1 1 0 1 0 0]
    4660
    




    array([False, False,  True, False,  True, False, False,  True,  True,
            True, False,  True,  True, False, False,  True], dtype=bool)



在这里，Biden和Obama被分到了不同的区域，那么和Obama在同一区域的文章，是不是就更类似Obama的呢？


```python
model['table'][model['bin_indices'][35817]]
```




    [21426, 35817, 39426, 50261, 53937]




```python
doc_ids = list(model['table'][model['bin_indices'][35817]])
doc_ids.remove(35817) # display documents other than Obama

docs = wiki.filter_by(values=doc_ids, column_name='id') # filter by id column
docs
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">id</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">URI</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">text</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">tf_idf</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">21426</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Mark_Boulware&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Mark Boulware</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">mark boulware born 1948<br>is an american diplomat ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'ambassador':<br>15.90834582606623, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">39426</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/John_Wells_(polit ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">John Wells (politician)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">sir john julius wells<br>born 30 march 1925 is a ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'when':<br>1.3806055739282235, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">50261</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Francis_Longstaff&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Francis Longstaff</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">francis a longstaff born<br>august 3 1956 is an ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'all':<br>1.6431112434912472, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">53937</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Madurai_T._Sriniv ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Madurai T. Srinivasan</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">maduraitsrinivasan is a<br>wellknown figure in the ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'alarmelu':<br>21.972990778450388, ...</td>
    </tr>
</table>
[4 rows x 5 columns]<br/>
</div>



实际上，Joe Biden与Obama的相似度比上面这四篇都要高。看一下它们的cosine distance


```python
def cosine_distance(x, y):
    xy = x.dot(y.T)
    dist = xy/(norm(x)*norm(y))
    return 1-dist[0,0]

obama_tf_idf = corpus[35817,:]
biden_tf_idf = corpus[24478,:]

print '================= Cosine distance from Barack Obama'
print 'Barack Obama - {0:24s}: {1:f}'.format('Joe Biden',
                                             cosine_distance(obama_tf_idf, biden_tf_idf))
for doc_id in doc_ids:
    doc_tf_idf = corpus[doc_id,:]
    print 'Barack Obama - {0:24s}: {1:f}'.format(wiki[doc_id]['name'],
                                                 cosine_distance(obama_tf_idf, doc_tf_idf))
```

    ================= Cosine distance from Barack Obama
    Barack Obama - Joe Biden               : 0.703139
    Barack Obama - Mark Boulware           : 0.950867
    Barack Obama - John Wells (politician) : 0.975966
    Barack Obama - Francis Longstaff       : 0.978256
    Barack Obama - Madurai T. Srinivasan   : 0.993092
    

**启示：**上述的观察结果说明，在LSH中相似的数据大体上是趋于落在相邻区域的，但是在高维空间中，相似点可能被随机向量分到不同的区域，而不相似的在同一区。**所以对一篇当前阅读文章查询，有必要考虑所有邻近区域，根据实际的距离排序。**

<h3 id = "半径">在训练的LSH模型上搜索 最近邻</h3>

首先，对LSH进行搜索，顺序是这样的：

1. L是query point所在区域的二进制指标。
2. 搜索所有在L中的文章。
3. 搜索所有和L有一个二进制位不同的区域中的文章。
4. 搜索所有和L有两个二进制位不同的区域中的文章。
...

为了获得所有和L有某些位不同的候选区域，我们用[itertools.combinations](https://docs.python.org/3/library/itertools.html#itertools.combinations "itertools.combinations"),来产生给定列表的所有可能子集。

1. 决定搜索半径r，即有多少位不同。
2. 对列[0,1,2,...,num_vector-1]的每一个子集[$n_1,n_2,...,n_r$]:
    + 改变query point的二进制指标的($n_1,n_2,...,n_r$)位，得到新的区域指标
    + 获得这个区域指标的文章列
    + 将这些文章添加到候选检查集合中
    
运行下面代码块，得到的每一行，是一个3元组，表示哪几个二进制位不同。


```python
from itertools import combinations

num_vector = 16
search_radius = 3

for diff in combinations(range(num_vector), search_radius):
    print diff
```

    (0, 1, 2)
    (0, 1, 3)
    (0, 1, 4)
    (0, 1, 5)
    (0, 1, 6)
    (0, 1, 7)
    (0, 1, 8)
    (0, 1, 9)
    (0, 1, 10)
    (0, 1, 11)
    (0, 1, 12)
    (0, 1, 13)
    (0, 1, 14)
    (0, 1, 15)
    (0, 2, 3)
    (0, 2, 4)
    (0, 2, 5)
    (0, 2, 6)
    (0, 2, 7)
    (0, 2, 8)
    (0, 2, 9)
    (0, 2, 10)
    (0, 2, 11)
    (0, 2, 12)
    (0, 2, 13)
    (0, 2, 14)
    (0, 2, 15)
    (0, 3, 4)
    (0, 3, 5)
    (0, 3, 6)
    (0, 3, 7)
    (0, 3, 8)
    (0, 3, 9)
    (0, 3, 10)
    (0, 3, 11)
    (0, 3, 12)
    (0, 3, 13)
    (0, 3, 14)
    (0, 3, 15)
    (0, 4, 5)
    (0, 4, 6)
    (0, 4, 7)
    (0, 4, 8)
    (0, 4, 9)
    (0, 4, 10)
    (0, 4, 11)
    (0, 4, 12)
    (0, 4, 13)
    (0, 4, 14)
    (0, 4, 15)
    (0, 5, 6)
    (0, 5, 7)
    (0, 5, 8)
    (0, 5, 9)
    (0, 5, 10)
    (0, 5, 11)
    (0, 5, 12)
    (0, 5, 13)
    (0, 5, 14)
    (0, 5, 15)
    (0, 6, 7)
    (0, 6, 8)
    (0, 6, 9)
    (0, 6, 10)
    (0, 6, 11)
    (0, 6, 12)
    (0, 6, 13)
    (0, 6, 14)
    (0, 6, 15)
    (0, 7, 8)
    (0, 7, 9)
    (0, 7, 10)
    (0, 7, 11)
    (0, 7, 12)
    (0, 7, 13)
    (0, 7, 14)
    (0, 7, 15)
    (0, 8, 9)
    (0, 8, 10)
    (0, 8, 11)
    (0, 8, 12)
    (0, 8, 13)
    (0, 8, 14)
    (0, 8, 15)
    (0, 9, 10)
    (0, 9, 11)
    (0, 9, 12)
    (0, 9, 13)
    (0, 9, 14)
    (0, 9, 15)
    (0, 10, 11)
    (0, 10, 12)
    (0, 10, 13)
    (0, 10, 14)
    (0, 10, 15)
    (0, 11, 12)
    (0, 11, 13)
    (0, 11, 14)
    (0, 11, 15)
    (0, 12, 13)
    (0, 12, 14)
    (0, 12, 15)
    (0, 13, 14)
    (0, 13, 15)
    (0, 14, 15)
    (1, 2, 3)
    (1, 2, 4)
    (1, 2, 5)
    (1, 2, 6)
    (1, 2, 7)
    (1, 2, 8)
    (1, 2, 9)
    (1, 2, 10)
    (1, 2, 11)
    (1, 2, 12)
    (1, 2, 13)
    (1, 2, 14)
    (1, 2, 15)
    (1, 3, 4)
    (1, 3, 5)
    (1, 3, 6)
    (1, 3, 7)
    (1, 3, 8)
    (1, 3, 9)
    (1, 3, 10)
    (1, 3, 11)
    (1, 3, 12)
    (1, 3, 13)
    (1, 3, 14)
    (1, 3, 15)
    (1, 4, 5)
    (1, 4, 6)
    (1, 4, 7)
    (1, 4, 8)
    (1, 4, 9)
    (1, 4, 10)
    (1, 4, 11)
    (1, 4, 12)
    (1, 4, 13)
    (1, 4, 14)
    (1, 4, 15)
    (1, 5, 6)
    (1, 5, 7)
    (1, 5, 8)
    (1, 5, 9)
    (1, 5, 10)
    (1, 5, 11)
    (1, 5, 12)
    (1, 5, 13)
    (1, 5, 14)
    (1, 5, 15)
    (1, 6, 7)
    (1, 6, 8)
    (1, 6, 9)
    (1, 6, 10)
    (1, 6, 11)
    (1, 6, 12)
    (1, 6, 13)
    (1, 6, 14)
    (1, 6, 15)
    (1, 7, 8)
    (1, 7, 9)
    (1, 7, 10)
    (1, 7, 11)
    (1, 7, 12)
    (1, 7, 13)
    (1, 7, 14)
    (1, 7, 15)
    (1, 8, 9)
    (1, 8, 10)
    (1, 8, 11)
    (1, 8, 12)
    (1, 8, 13)
    (1, 8, 14)
    (1, 8, 15)
    (1, 9, 10)
    (1, 9, 11)
    (1, 9, 12)
    (1, 9, 13)
    (1, 9, 14)
    (1, 9, 15)
    (1, 10, 11)
    (1, 10, 12)
    (1, 10, 13)
    (1, 10, 14)
    (1, 10, 15)
    (1, 11, 12)
    (1, 11, 13)
    (1, 11, 14)
    (1, 11, 15)
    (1, 12, 13)
    (1, 12, 14)
    (1, 12, 15)
    (1, 13, 14)
    (1, 13, 15)
    (1, 14, 15)
    (2, 3, 4)
    (2, 3, 5)
    (2, 3, 6)
    (2, 3, 7)
    (2, 3, 8)
    (2, 3, 9)
    (2, 3, 10)
    (2, 3, 11)
    (2, 3, 12)
    (2, 3, 13)
    (2, 3, 14)
    (2, 3, 15)
    (2, 4, 5)
    (2, 4, 6)
    (2, 4, 7)
    (2, 4, 8)
    (2, 4, 9)
    (2, 4, 10)
    (2, 4, 11)
    (2, 4, 12)
    (2, 4, 13)
    (2, 4, 14)
    (2, 4, 15)
    (2, 5, 6)
    (2, 5, 7)
    (2, 5, 8)
    (2, 5, 9)
    (2, 5, 10)
    (2, 5, 11)
    (2, 5, 12)
    (2, 5, 13)
    (2, 5, 14)
    (2, 5, 15)
    (2, 6, 7)
    (2, 6, 8)
    (2, 6, 9)
    (2, 6, 10)
    (2, 6, 11)
    (2, 6, 12)
    (2, 6, 13)
    (2, 6, 14)
    (2, 6, 15)
    (2, 7, 8)
    (2, 7, 9)
    (2, 7, 10)
    (2, 7, 11)
    (2, 7, 12)
    (2, 7, 13)
    (2, 7, 14)
    (2, 7, 15)
    (2, 8, 9)
    (2, 8, 10)
    (2, 8, 11)
    (2, 8, 12)
    (2, 8, 13)
    (2, 8, 14)
    (2, 8, 15)
    (2, 9, 10)
    (2, 9, 11)
    (2, 9, 12)
    (2, 9, 13)
    (2, 9, 14)
    (2, 9, 15)
    (2, 10, 11)
    (2, 10, 12)
    (2, 10, 13)
    (2, 10, 14)
    (2, 10, 15)
    (2, 11, 12)
    (2, 11, 13)
    (2, 11, 14)
    (2, 11, 15)
    (2, 12, 13)
    (2, 12, 14)
    (2, 12, 15)
    (2, 13, 14)
    (2, 13, 15)
    (2, 14, 15)
    (3, 4, 5)
    (3, 4, 6)
    (3, 4, 7)
    (3, 4, 8)
    (3, 4, 9)
    (3, 4, 10)
    (3, 4, 11)
    (3, 4, 12)
    (3, 4, 13)
    (3, 4, 14)
    (3, 4, 15)
    (3, 5, 6)
    (3, 5, 7)
    (3, 5, 8)
    (3, 5, 9)
    (3, 5, 10)
    (3, 5, 11)
    (3, 5, 12)
    (3, 5, 13)
    (3, 5, 14)
    (3, 5, 15)
    (3, 6, 7)
    (3, 6, 8)
    (3, 6, 9)
    (3, 6, 10)
    (3, 6, 11)
    (3, 6, 12)
    (3, 6, 13)
    (3, 6, 14)
    (3, 6, 15)
    (3, 7, 8)
    (3, 7, 9)
    (3, 7, 10)
    (3, 7, 11)
    (3, 7, 12)
    (3, 7, 13)
    (3, 7, 14)
    (3, 7, 15)
    (3, 8, 9)
    (3, 8, 10)
    (3, 8, 11)
    (3, 8, 12)
    (3, 8, 13)
    (3, 8, 14)
    (3, 8, 15)
    (3, 9, 10)
    (3, 9, 11)
    (3, 9, 12)
    (3, 9, 13)
    (3, 9, 14)
    (3, 9, 15)
    (3, 10, 11)
    (3, 10, 12)
    (3, 10, 13)
    (3, 10, 14)
    (3, 10, 15)
    (3, 11, 12)
    (3, 11, 13)
    (3, 11, 14)
    (3, 11, 15)
    (3, 12, 13)
    (3, 12, 14)
    (3, 12, 15)
    (3, 13, 14)
    (3, 13, 15)
    (3, 14, 15)
    (4, 5, 6)
    (4, 5, 7)
    (4, 5, 8)
    (4, 5, 9)
    (4, 5, 10)
    (4, 5, 11)
    (4, 5, 12)
    (4, 5, 13)
    (4, 5, 14)
    (4, 5, 15)
    (4, 6, 7)
    (4, 6, 8)
    (4, 6, 9)
    (4, 6, 10)
    (4, 6, 11)
    (4, 6, 12)
    (4, 6, 13)
    (4, 6, 14)
    (4, 6, 15)
    (4, 7, 8)
    (4, 7, 9)
    (4, 7, 10)
    (4, 7, 11)
    (4, 7, 12)
    (4, 7, 13)
    (4, 7, 14)
    (4, 7, 15)
    (4, 8, 9)
    (4, 8, 10)
    (4, 8, 11)
    (4, 8, 12)
    (4, 8, 13)
    (4, 8, 14)
    (4, 8, 15)
    (4, 9, 10)
    (4, 9, 11)
    (4, 9, 12)
    (4, 9, 13)
    (4, 9, 14)
    (4, 9, 15)
    (4, 10, 11)
    (4, 10, 12)
    (4, 10, 13)
    (4, 10, 14)
    (4, 10, 15)
    (4, 11, 12)
    (4, 11, 13)
    (4, 11, 14)
    (4, 11, 15)
    (4, 12, 13)
    (4, 12, 14)
    (4, 12, 15)
    (4, 13, 14)
    (4, 13, 15)
    (4, 14, 15)
    (5, 6, 7)
    (5, 6, 8)
    (5, 6, 9)
    (5, 6, 10)
    (5, 6, 11)
    (5, 6, 12)
    (5, 6, 13)
    (5, 6, 14)
    (5, 6, 15)
    (5, 7, 8)
    (5, 7, 9)
    (5, 7, 10)
    (5, 7, 11)
    (5, 7, 12)
    (5, 7, 13)
    (5, 7, 14)
    (5, 7, 15)
    (5, 8, 9)
    (5, 8, 10)
    (5, 8, 11)
    (5, 8, 12)
    (5, 8, 13)
    (5, 8, 14)
    (5, 8, 15)
    (5, 9, 10)
    (5, 9, 11)
    (5, 9, 12)
    (5, 9, 13)
    (5, 9, 14)
    (5, 9, 15)
    (5, 10, 11)
    (5, 10, 12)
    (5, 10, 13)
    (5, 10, 14)
    (5, 10, 15)
    (5, 11, 12)
    (5, 11, 13)
    (5, 11, 14)
    (5, 11, 15)
    (5, 12, 13)
    (5, 12, 14)
    (5, 12, 15)
    (5, 13, 14)
    (5, 13, 15)
    (5, 14, 15)
    (6, 7, 8)
    (6, 7, 9)
    (6, 7, 10)
    (6, 7, 11)
    (6, 7, 12)
    (6, 7, 13)
    (6, 7, 14)
    (6, 7, 15)
    (6, 8, 9)
    (6, 8, 10)
    (6, 8, 11)
    (6, 8, 12)
    (6, 8, 13)
    (6, 8, 14)
    (6, 8, 15)
    (6, 9, 10)
    (6, 9, 11)
    (6, 9, 12)
    (6, 9, 13)
    (6, 9, 14)
    (6, 9, 15)
    (6, 10, 11)
    (6, 10, 12)
    (6, 10, 13)
    (6, 10, 14)
    (6, 10, 15)
    (6, 11, 12)
    (6, 11, 13)
    (6, 11, 14)
    (6, 11, 15)
    (6, 12, 13)
    (6, 12, 14)
    (6, 12, 15)
    (6, 13, 14)
    (6, 13, 15)
    (6, 14, 15)
    (7, 8, 9)
    (7, 8, 10)
    (7, 8, 11)
    (7, 8, 12)
    (7, 8, 13)
    (7, 8, 14)
    (7, 8, 15)
    (7, 9, 10)
    (7, 9, 11)
    (7, 9, 12)
    (7, 9, 13)
    (7, 9, 14)
    (7, 9, 15)
    (7, 10, 11)
    (7, 10, 12)
    (7, 10, 13)
    (7, 10, 14)
    (7, 10, 15)
    (7, 11, 12)
    (7, 11, 13)
    (7, 11, 14)
    (7, 11, 15)
    (7, 12, 13)
    (7, 12, 14)
    (7, 12, 15)
    (7, 13, 14)
    (7, 13, 15)
    (7, 14, 15)
    (8, 9, 10)
    (8, 9, 11)
    (8, 9, 12)
    (8, 9, 13)
    (8, 9, 14)
    (8, 9, 15)
    (8, 10, 11)
    (8, 10, 12)
    (8, 10, 13)
    (8, 10, 14)
    (8, 10, 15)
    (8, 11, 12)
    (8, 11, 13)
    (8, 11, 14)
    (8, 11, 15)
    (8, 12, 13)
    (8, 12, 14)
    (8, 12, 15)
    (8, 13, 14)
    (8, 13, 15)
    (8, 14, 15)
    (9, 10, 11)
    (9, 10, 12)
    (9, 10, 13)
    (9, 10, 14)
    (9, 10, 15)
    (9, 11, 12)
    (9, 11, 13)
    (9, 11, 14)
    (9, 11, 15)
    (9, 12, 13)
    (9, 12, 14)
    (9, 12, 15)
    (9, 13, 14)
    (9, 13, 15)
    (9, 14, 15)
    (10, 11, 12)
    (10, 11, 13)
    (10, 11, 14)
    (10, 11, 15)
    (10, 12, 13)
    (10, 12, 14)
    (10, 12, 15)
    (10, 13, 14)
    (10, 13, 15)
    (10, 14, 15)
    (11, 12, 13)
    (11, 12, 14)
    (11, 12, 15)
    (11, 13, 14)
    (11, 13, 15)
    (11, 14, 15)
    (12, 13, 14)
    (12, 13, 15)
    (12, 14, 15)
    (13, 14, 15)
    

下面实现近邻搜索：


```python
def search_nearby_bins(query_bin_bits, table, search_radius=2, initial_candidates=set()):
    """
    For a given query vector and trained LSH model, return all candidate neighbors for
    the query among all bins within the given search radius.
    
    Example usage
    -------------
    >>> model = train_lsh(corpus, num_vector=16, seed=143)
    >>> q = model['bin_index_bits'][0]  # vector for the first document
  
    >>> candidates = search_nearby_bins(q, model['table'])
    """
    num_vector = len(query_bin_bits)
    powers_of_two = 1 << np.arange(num_vector-1, -1, -1)
    
    # Allow the user to provide an initial set of candidates.
    candidate_set = copy(initial_candidates)
    
    for different_bits in combinations(range(num_vector), search_radius):       
        # Flip the bits (n_1,n_2,...,n_r) of the query bin to produce a new bit vector.
        ## Hint: you can iterate over a tuple like a list
        alternate_bits = copy(query_bin_bits)
        for i in different_bits:
            alternate_bits[i] = 1 - query_bin_bits[i] # YOUR CODE HERE 
        
        # Convert the new bit vector to an integer index
        nearby_bin = alternate_bits.dot(powers_of_two)
        
        # Fetch the list of documents belonging to the bin indexed by the new bit vector.
        # Then add those documents to candidate_set
        # Make sure that the bin exists in the table!
        # Hint: update() method for sets lets you add an entire list to the set
        if nearby_bin in table:
            candidate_set.update(table[nearby_bin]) # YOUR CODE HERE: Update candidate_set with the documents in this bin.
            
    return candidate_set
```

用search_radius=0测试一下，应该得到和query point在一个区域的文章列。


```python
obama_bin_index = model['bin_index_bits'][35817] # bin index of Barack Obama
candidate_set = search_nearby_bins(obama_bin_index, model['table'], search_radius=0)
if candidate_set == set([35817, 21426, 53937, 39426, 50261]):
    print 'Passed test'
else:
    print 'Check your code'
print 'List of documents in the same bin as Obama: 35817, 21426, 53937, 39426, 50261'
```

    Passed test
    List of documents in the same bin as Obama: 35817, 21426, 53937, 39426, 50261
    

用search_radius=1，应该有增加文章。


```python
candidate_set = search_nearby_bins(obama_bin_index, model['table'], search_radius=1, initial_candidates=candidate_set)
if candidate_set == set([39426, 38155, 38412, 28444, 9757, 41631, 39207, 59050, 47773, 53937, 21426, 34547,
                         23229, 55615, 39877, 27404, 33996, 21715, 50261, 21975, 33243, 58723, 35817, 45676,
                         19699, 2804, 20347]):
    print 'Passed test'
else:
    print 'Check your code'
```

    Passed test
    

**注意：**这里得到的和Obama相似的文章很少，所以我们要尽可能地增加搜索半径，以获得更多的候选。

现在，我们就能利用这些候选文章列，来进行最近邻搜索。


```python
def query(vec, model, k, max_search_radius):
  
    data = model['data']
    table = model['table']
    random_vectors = model['random_vectors']
    num_vector = random_vectors.shape[1]
    
    
    # Compute bin index for the query vector, in bit representation.
    bin_index_bits = (vec.dot(random_vectors) >= 0).flatten()
    
    # Search nearby bins and collect candidates
    candidate_set = set()
    for search_radius in xrange(max_search_radius+1):
        candidate_set = search_nearby_bins(bin_index_bits, table, search_radius, initial_candidates=candidate_set)
    
    # Sort candidates by their true distances from the query
    nearest_neighbors = graphlab.SFrame({'id':candidate_set})
    candidates = data[np.array(list(candidate_set)),:]
    nearest_neighbors['distance'] = pairwise_distances(candidates, vec, metric='cosine').flatten()
    
    return nearest_neighbors.topk('distance', k, reverse=True), len(candidate_set)
```

用Obama来测试一下，最大搜索半径为3。


```python
query(corpus[35817,:], model, k=10, max_search_radius=3)
```




    (Columns:
     	id	int
     	distance	float
     
     Rows: 10
     
     Data:
     +-------+----------------+
     |   id  |    distance    |
     +-------+----------------+
     | 35817 |      0.0       |
     | 24478 | 0.703138676734 |
     | 56008 | 0.856848127628 |
     | 37199 | 0.874668698194 |
     | 40353 | 0.890034225981 |
     |  9267 | 0.898377208819 |
     | 55909 | 0.899340396322 |
     |  9165 | 0.900921029925 |
     | 57958 | 0.903003263483 |
     | 49872 | 0.909532800353 |
     +-------+----------------+
     [10 rows x 2 columns], 727)




```python
query(corpus[35817,:], model, k=10, max_search_radius=3)[0].join(wiki[['id', 'name']], on='id').sort('distance')
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">id</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">distance</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">35817</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">24478</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.703138676734</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Joe Biden</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">56008</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.856848127628</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Nathan Cullen</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">37199</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.874668698194</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barry Sullivan (lawyer)</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">40353</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.890034225981</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Neil MacBride</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9267</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.898377208819</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Vikramaditya Khanna</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">55909</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.899340396322</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Herman Cain</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9165</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.900921029925</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Raymond F. Clevenger</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">57958</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.903003263483</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Michael J. Malbin</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">49872</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.909532800353</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Lowell Barron</td>
    </tr>
</table>
[10 rows x 3 columns]<br/>
</div>



### 邻近区域搜索对结果的影响

由于搜索半径的不同，有三个直观的影响：

+ 候选检查文章的数目
+ 搜索时间
+ 近似近邻到当前文章的距离

对不同的搜索半径进行实验，比较这三者的变化。


```python
wiki[wiki['name']=='Barack Obama'] # still use Obama as query point
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">id</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">URI</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">text</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">tf_idf</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">35817</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">&lt;http://dbpedia.org/resou<br>rce/Barack_Obama&gt; ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Barack Obama</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">barack hussein obama ii<br>brk husen bm born august ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'operations':<br>3.811771079388818, ...</td>
    </tr>
</table>
[? rows x 5 columns]<br/>Note: Only the head of the SFrame is printed. This SFrame is lazily evaluated.<br/>You can use sf.materialize() to force materialization.
</div>




```python
num_candidates_history = []
query_time_history = []
max_distance_from_query_history = []
min_distance_from_query_history = []
average_distance_from_query_history = []

for max_search_radius in xrange(17):
    start=time.time()
    result, num_candidates = query(corpus[35817,:], model, k=10,
                                   max_search_radius=max_search_radius)
    end=time.time()
    query_time = end-start
    
    print 'Radius:', max_search_radius
    print result.join(wiki[['id', 'name']], on='id').sort('distance')
    
    average_distance_from_query = result['distance'][1:].mean()
    max_distance_from_query = result['distance'][1:].max()
    min_distance_from_query = result['distance'][1:].min()
    
    num_candidates_history.append(num_candidates)
    query_time_history.append(query_time)
    average_distance_from_query_history.append(average_distance_from_query)
    max_distance_from_query_history.append(max_distance_from_query)
    min_distance_from_query_history.append(min_distance_from_query)
```

    Radius: 0
    +-------+----------------+-------------------------+
    |   id  |    distance    |           name          |
    +-------+----------------+-------------------------+
    | 35817 |      0.0       |       Barack Obama      |
    | 21426 | 0.950866757525 |      Mark Boulware      |
    | 39426 | 0.97596600411  | John Wells (politician) |
    | 50261 | 0.978256163041 |    Francis Longstaff    |
    | 53937 | 0.993092148424 |  Madurai T. Srinivasan  |
    +-------+----------------+-------------------------+
    [5 rows x 3 columns]
    
    Radius: 1
    +-------+----------------+-------------------------------+
    |   id  |    distance    |              name             |
    +-------+----------------+-------------------------------+
    | 35817 |      0.0       |          Barack Obama         |
    | 41631 | 0.947459482005 |          Binayak Sen          |
    | 21426 | 0.950866757525 |         Mark Boulware         |
    | 33243 | 0.951765770113 |        Janice Lachance        |
    | 33996 | 0.960859054157 |          Rufus Black          |
    | 28444 | 0.961080585824 |        John Paul Phelan       |
    | 20347 | 0.974129605472 |        Gianni De Fraja        |
    | 39426 | 0.97596600411  |    John Wells (politician)    |
    | 34547 | 0.978214931987 | Nathan Murphy (Australian ... |
    | 50261 | 0.978256163041 |       Francis Longstaff       |
    +-------+----------------+-------------------------------+
    [10 rows x 3 columns]
    
    Radius: 2
    +-------+----------------+---------------------+
    |   id  |    distance    |         name        |
    +-------+----------------+---------------------+
    | 35817 |      0.0       |     Barack Obama    |
    | 24478 | 0.703138676734 |      Joe Biden      |
    |  9267 | 0.898377208819 | Vikramaditya Khanna |
    | 55909 | 0.899340396322 |     Herman Cain     |
    |  6949 | 0.925713001103 |  Harrison J. Goldin |
    | 23524 | 0.926397988994 |    Paul Bennecke    |
    |  5823 | 0.928498260316 |    Adeleke Mamora   |
    | 37262 | 0.93445433211  |      Becky Cain     |
    | 10121 | 0.936896394645 |     Bill Bradley    |
    | 54782 | 0.937809202206 |  Thomas F. Hartnett |
    +-------+----------------+---------------------+
    [10 rows x 3 columns]
    
    Radius: 3
    +-------+----------------+-------------------------+
    |   id  |    distance    |           name          |
    +-------+----------------+-------------------------+
    | 35817 |      0.0       |       Barack Obama      |
    | 24478 | 0.703138676734 |        Joe Biden        |
    | 56008 | 0.856848127628 |      Nathan Cullen      |
    | 37199 | 0.874668698194 | Barry Sullivan (lawyer) |
    | 40353 | 0.890034225981 |      Neil MacBride      |
    |  9267 | 0.898377208819 |   Vikramaditya Khanna   |
    | 55909 | 0.899340396322 |       Herman Cain       |
    |  9165 | 0.900921029925 |   Raymond F. Clevenger  |
    | 57958 | 0.903003263483 |    Michael J. Malbin    |
    | 49872 | 0.909532800353 |      Lowell Barron      |
    +-------+----------------+-------------------------+
    [10 rows x 3 columns]
    
    Radius: 4
    +-------+----------------+--------------------+
    |   id  |    distance    |        name        |
    +-------+----------------+--------------------+
    | 35817 |      0.0       |    Barack Obama    |
    | 24478 | 0.703138676734 |     Joe Biden      |
    | 36452 | 0.833985493688 |    Bill Clinton    |
    | 24848 | 0.839406735668 |  John C. Eastman   |
    | 43155 | 0.840839007484 |    Goodwin Liu     |
    | 42965 | 0.849077676943 |  John O. Brennan   |
    | 56008 | 0.856848127628 |   Nathan Cullen    |
    | 38495 | 0.857573828556 |    Barney Frank    |
    | 18752 | 0.858899032522 |   Dan W. Reicher   |
    |  2092 | 0.874643264756 | Richard Blumenthal |
    +-------+----------------+--------------------+
    [10 rows x 3 columns]
    
    Radius: 5
    +-------+----------------+-------------------------+
    |   id  |    distance    |           name          |
    +-------+----------------+-------------------------+
    | 35817 |      0.0       |       Barack Obama      |
    | 24478 | 0.703138676734 |        Joe Biden        |
    | 38714 | 0.770561227601 | Eric Stern (politician) |
    | 46811 | 0.800197384104 |      Jeff Sessions      |
    | 14754 | 0.826854025897 |       Mitt Romney       |
    | 36452 | 0.833985493688 |       Bill Clinton      |
    | 40943 | 0.834534928232 |      Jonathan Alter     |
    | 55044 | 0.837013236281 |       Wesley Clark      |
    | 24848 | 0.839406735668 |     John C. Eastman     |
    | 43155 | 0.840839007484 |       Goodwin Liu       |
    +-------+----------------+-------------------------+
    [10 rows x 3 columns]
    
    Radius: 6
    +-------+----------------+-------------------------+
    |   id  |    distance    |           name          |
    +-------+----------------+-------------------------+
    | 35817 |      0.0       |       Barack Obama      |
    | 24478 | 0.703138676734 |        Joe Biden        |
    | 38714 | 0.770561227601 | Eric Stern (politician) |
    | 44681 | 0.790926415366 |  Jesse Lee (politician) |
    | 46811 | 0.800197384104 |      Jeff Sessions      |
    | 48693 | 0.809192212293 |       Artur Davis       |
    | 23737 | 0.810164633465 |    John D. McCormick    |
    |  4032 | 0.814554748671 |   Kenneth D. Thompson   |
    | 28447 | 0.823228984384 |      George W. Bush     |
    | 14754 | 0.826854025897 |       Mitt Romney       |
    +-------+----------------+-------------------------+
    [10 rows x 3 columns]
    
    Radius: 7
    +-------+----------------+-------------------------+
    |   id  |    distance    |           name          |
    +-------+----------------+-------------------------+
    | 35817 |      0.0       |       Barack Obama      |
    | 24478 | 0.703138676734 |        Joe Biden        |
    | 38376 | 0.742981902328 |      Samantha Power     |
    | 57108 | 0.758358397887 |  Hillary Rodham Clinton |
    | 38714 | 0.770561227601 | Eric Stern (politician) |
    | 44681 | 0.790926415366 |  Jesse Lee (politician) |
    | 18827 | 0.798322602893 |       Henry Waxman      |
    | 46811 | 0.800197384104 |      Jeff Sessions      |
    | 48693 | 0.809192212293 |       Artur Davis       |
    | 23737 | 0.810164633465 |    John D. McCormick    |
    +-------+----------------+-------------------------+
    [10 rows x 3 columns]
    
    Radius: 8
    +-------+----------------+-------------------------+
    |   id  |    distance    |           name          |
    +-------+----------------+-------------------------+
    | 35817 |      0.0       |       Barack Obama      |
    | 24478 | 0.703138676734 |        Joe Biden        |
    | 38376 | 0.742981902328 |      Samantha Power     |
    | 57108 | 0.758358397887 |  Hillary Rodham Clinton |
    | 38714 | 0.770561227601 | Eric Stern (politician) |
    | 44681 | 0.790926415366 |  Jesse Lee (politician) |
    | 18827 | 0.798322602893 |       Henry Waxman      |
    | 46811 | 0.800197384104 |      Jeff Sessions      |
    | 48693 | 0.809192212293 |       Artur Davis       |
    | 23737 | 0.810164633465 |    John D. McCormick    |
    +-------+----------------+-------------------------+
    [10 rows x 3 columns]
    
    Radius: 9
    +-------+----------------+-------------------------+
    |   id  |    distance    |           name          |
    +-------+----------------+-------------------------+
    | 35817 |      0.0       |       Barack Obama      |
    | 24478 | 0.703138676734 |        Joe Biden        |
    | 38376 | 0.742981902328 |      Samantha Power     |
    | 57108 | 0.758358397887 |  Hillary Rodham Clinton |
    | 38714 | 0.770561227601 | Eric Stern (politician) |
    | 46140 | 0.784677504751 |       Robert Gibbs      |
    | 44681 | 0.790926415366 |  Jesse Lee (politician) |
    | 18827 | 0.798322602893 |       Henry Waxman      |
    | 46811 | 0.800197384104 |      Jeff Sessions      |
    | 39357 | 0.809050776238 |       John McCain       |
    +-------+----------------+-------------------------+
    [10 rows x 3 columns]
    
    Radius: 10
    +-------+----------------+-------------------------+
    |   id  |    distance    |           name          |
    +-------+----------------+-------------------------+
    | 35817 |      0.0       |       Barack Obama      |
    | 24478 | 0.703138676734 |        Joe Biden        |
    | 38376 | 0.742981902328 |      Samantha Power     |
    | 57108 | 0.758358397887 |  Hillary Rodham Clinton |
    | 38714 | 0.770561227601 | Eric Stern (politician) |
    | 46140 | 0.784677504751 |       Robert Gibbs      |
    | 44681 | 0.790926415366 |  Jesse Lee (politician) |
    | 18827 | 0.798322602893 |       Henry Waxman      |
    |  2412 | 0.799466360042 |     Joe the Plumber     |
    | 46811 | 0.800197384104 |      Jeff Sessions      |
    +-------+----------------+-------------------------+
    [10 rows x 3 columns]
    
    Radius: 11
    +-------+----------------+-------------------------+
    |   id  |    distance    |           name          |
    +-------+----------------+-------------------------+
    | 35817 |      0.0       |       Barack Obama      |
    | 24478 | 0.703138676734 |        Joe Biden        |
    | 38376 | 0.742981902328 |      Samantha Power     |
    | 57108 | 0.758358397887 |  Hillary Rodham Clinton |
    | 38714 | 0.770561227601 | Eric Stern (politician) |
    | 46140 | 0.784677504751 |       Robert Gibbs      |
    | 44681 | 0.790926415366 |  Jesse Lee (politician) |
    | 18827 | 0.798322602893 |       Henry Waxman      |
    |  2412 | 0.799466360042 |     Joe the Plumber     |
    | 46811 | 0.800197384104 |      Jeff Sessions      |
    +-------+----------------+-------------------------+
    [10 rows x 3 columns]
    
    Radius: 12
    +-------+----------------+-------------------------+
    |   id  |    distance    |           name          |
    +-------+----------------+-------------------------+
    | 35817 |      0.0       |       Barack Obama      |
    | 24478 | 0.703138676734 |        Joe Biden        |
    | 38376 | 0.742981902328 |      Samantha Power     |
    | 57108 | 0.758358397887 |  Hillary Rodham Clinton |
    | 38714 | 0.770561227601 | Eric Stern (politician) |
    | 46140 | 0.784677504751 |       Robert Gibbs      |
    |  6796 | 0.788039072943 |       Eric Holder       |
    | 44681 | 0.790926415366 |  Jesse Lee (politician) |
    | 18827 | 0.798322602893 |       Henry Waxman      |
    |  2412 | 0.799466360042 |     Joe the Plumber     |
    +-------+----------------+-------------------------+
    [10 rows x 3 columns]
    
    Radius: 13
    +-------+----------------+-------------------------+
    |   id  |    distance    |           name          |
    +-------+----------------+-------------------------+
    | 35817 |      0.0       |       Barack Obama      |
    | 24478 | 0.703138676734 |        Joe Biden        |
    | 38376 | 0.742981902328 |      Samantha Power     |
    | 57108 | 0.758358397887 |  Hillary Rodham Clinton |
    | 38714 | 0.770561227601 | Eric Stern (politician) |
    | 46140 | 0.784677504751 |       Robert Gibbs      |
    |  6796 | 0.788039072943 |       Eric Holder       |
    | 44681 | 0.790926415366 |  Jesse Lee (politician) |
    | 18827 | 0.798322602893 |       Henry Waxman      |
    |  2412 | 0.799466360042 |     Joe the Plumber     |
    +-------+----------------+-------------------------+
    [10 rows x 3 columns]
    
    Radius: 14
    +-------+----------------+-------------------------+
    |   id  |    distance    |           name          |
    +-------+----------------+-------------------------+
    | 35817 |      0.0       |       Barack Obama      |
    | 24478 | 0.703138676734 |        Joe Biden        |
    | 38376 | 0.742981902328 |      Samantha Power     |
    | 57108 | 0.758358397887 |  Hillary Rodham Clinton |
    | 38714 | 0.770561227601 | Eric Stern (politician) |
    | 46140 | 0.784677504751 |       Robert Gibbs      |
    |  6796 | 0.788039072943 |       Eric Holder       |
    | 44681 | 0.790926415366 |  Jesse Lee (politician) |
    | 18827 | 0.798322602893 |       Henry Waxman      |
    |  2412 | 0.799466360042 |     Joe the Plumber     |
    +-------+----------------+-------------------------+
    [10 rows x 3 columns]
    
    Radius: 15
    +-------+----------------+-------------------------+
    |   id  |    distance    |           name          |
    +-------+----------------+-------------------------+
    | 35817 |      0.0       |       Barack Obama      |
    | 24478 | 0.703138676734 |        Joe Biden        |
    | 38376 | 0.742981902328 |      Samantha Power     |
    | 57108 | 0.758358397887 |  Hillary Rodham Clinton |
    | 38714 | 0.770561227601 | Eric Stern (politician) |
    | 46140 | 0.784677504751 |       Robert Gibbs      |
    |  6796 | 0.788039072943 |       Eric Holder       |
    | 44681 | 0.790926415366 |  Jesse Lee (politician) |
    | 18827 | 0.798322602893 |       Henry Waxman      |
    |  2412 | 0.799466360042 |     Joe the Plumber     |
    +-------+----------------+-------------------------+
    [10 rows x 3 columns]
    
    Radius: 16
    +-------+----------------+-------------------------+
    |   id  |    distance    |           name          |
    +-------+----------------+-------------------------+
    | 35817 |      0.0       |       Barack Obama      |
    | 24478 | 0.703138676734 |        Joe Biden        |
    | 38376 | 0.742981902328 |      Samantha Power     |
    | 57108 | 0.758358397887 |  Hillary Rodham Clinton |
    | 38714 | 0.770561227601 | Eric Stern (politician) |
    | 46140 | 0.784677504751 |       Robert Gibbs      |
    |  6796 | 0.788039072943 |       Eric Holder       |
    | 44681 | 0.790926415366 |  Jesse Lee (politician) |
    | 18827 | 0.798322602893 |       Henry Waxman      |
    |  2412 | 0.799466360042 |     Joe the Plumber     |
    +-------+----------------+-------------------------+
    [10 rows x 3 columns]
    
    

为上述三个变量画图，观察到搜索结果中最相近的10篇文章随着搜索半径增加，越来越相关:


```python
plt.figure(figsize=(7,4.5))
plt.plot(num_candidates_history, linewidth=4)
plt.xlabel('Search radius')
plt.ylabel('# of documents searched')
plt.rcParams.update({'font.size':16})
plt.tight_layout()

plt.figure(figsize=(7,4.5))
plt.plot(query_time_history, linewidth=4)
plt.xlabel('Search radius')
plt.ylabel('Query time (seconds)')
plt.rcParams.update({'font.size':16})
plt.tight_layout()

plt.figure(figsize=(7,4.5))
plt.plot(average_distance_from_query_history, linewidth=4, label='Average of 10 neighbors')
plt.plot(max_distance_from_query_history, linewidth=4, label='Farthest of 10 neighbors')
plt.plot(min_distance_from_query_history, linewidth=4, label='Closest of 10 neighbors')
plt.xlabel('Search radius')
plt.ylabel('Cosine distance of neighbors')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size':16})
plt.tight_layout()
```


![png](output/output_220_0.png)



![png](output/output_220_1.png)



![png](output/output_220_2.png)


观察到的部分结果：

+ 随着搜索半径增加，找到更多的相似文章具有更小的距离。
+ 随着搜索半径增加，要检查的文章变多，相应的搜索时间更长。
+ 当搜索半径足够大时，LSH的搜索结果接近brute force搜索的结果。

**小问题：**如果希望得到的10篇近似最近邻文章，它们到query point的平均距离和真实的平均距离相差不差过0.01。比如，对Obama的真实数据时0.77，那么使搜索结果比0.78更好的最小搜索半径是多少？


```python
for distance in average_distance_from_query_history:
    if abs(distance - 0.77) <= 0.01:
        print average_distance_from_query_history.index(distance), distance
        break
```

    7 0.775982605852
    

### 近邻的准确性

为了测试是否普遍可靠，我们随机选10篇文章作为query point来分析。

对每篇文章，计算它真实的25最近邻，然后用LSH搜索，用两个标准来衡量：

+ 10篇中的准确率：LSH给出的10篇近邻中，有多少是在真正的25篇中的。
+ 到query的平均cosine distance

然后，用不同的搜索半径运行LSH多次。


```python
def brute_force_query(vec, data, k):
    num_data_points = data.shape[0]
    
    # Compute distances for ALL data points in training set
    nearest_neighbors = graphlab.SFrame({'id':range(num_data_points)})
    nearest_neighbors['distance'] = pairwise_distances(data, vec, metric='cosine').flatten()
    
    return nearest_neighbors.topk('distance', k, reverse=True)
```

下面的代码块，对不同的搜索半径运行LSH，并测量两个标准(可能要运行一段时间)：


```python
max_radius = 17
precision = {i:[] for i in xrange(max_radius)}
average_distance  = {i:[] for i in xrange(max_radius)}
query_time  = {i:[] for i in xrange(max_radius)}

np.random.seed(0)
num_queries = 10
for i, ix in enumerate(np.random.choice(corpus.shape[0], num_queries, replace=False)):
    print('%s / %s' % (i, num_queries))
    ground_truth = set(brute_force_query(corpus[ix,:], corpus, k=25)['id'])
    # Get the set of 25 true nearest neighbors
    
    for r in xrange(1,max_radius):
        start = time.time()
        result, num_candidates = query(corpus[ix,:], model, k=10, max_search_radius=r)
        end = time.time()

        query_time[r].append(end-start)
        # precision = (# of neighbors both in result and ground_truth)/10.0
        precision[r].append(len(set(result['id']) & ground_truth)/10.0)
        average_distance[r].append(result['distance'][1:].mean())
```

    0 / 10
    1 / 10
    2 / 10
    3 / 10
    4 / 10
    5 / 10
    6 / 10
    7 / 10
    8 / 10
    9 / 10
    


```python
plt.figure(figsize=(7,4.5))
plt.plot(range(1,17), [np.mean(average_distance[i]) for i in xrange(1,17)], linewidth=4, label='Average over 10 neighbors')
plt.xlabel('Search radius')
plt.ylabel('Cosine distance')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size':16})
plt.tight_layout()

plt.figure(figsize=(7,4.5))
plt.plot(range(1,17), [np.mean(precision[i]) for i in xrange(1,17)], linewidth=4, label='Precison@10')
plt.xlabel('Search radius')
plt.ylabel('Precision')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size':16})
plt.tight_layout()

plt.figure(figsize=(7,4.5))
plt.plot(range(1,17), [np.mean(query_time[i]) for i in xrange(1,17)], linewidth=4, label='Query time')
plt.xlabel('Search radius')
plt.ylabel('Query time (seconds)')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size':16})
plt.tight_layout()
```


![png](output/output_229_0.png)



![png](output/output_229_1.png)



![png](output/output_229_2.png)


发现Obama作为query point的观察结果可以推广到整个数据集上。

<h3 id = "向量">随机向量的个数的影响</h3>

接下来，我们看剩下的这个变量，随机向量的个数，对LSH效果的影响。固定搜索半径：3，对5-20个随机向量，运行LSH。要花一段时间。


```python
precision = {i:[] for i in xrange(5,20)}
average_distance  = {i:[] for i in xrange(5,20)}
query_time = {i:[] for i in xrange(5,20)}
num_candidates_history = {i:[] for i in xrange(5,20)}
ground_truth = {}

np.random.seed(0)
num_queries = 10
docs = np.random.choice(corpus.shape[0], num_queries, replace=False)

for i, ix in enumerate(docs):
    ground_truth[ix] = set(brute_force_query(corpus[ix,:], corpus, k=25)['id'])
    # Get the set of 25 true nearest neighbors

for num_vector in xrange(5,20):
    print('num_vector = %s' % (num_vector))
    model = train_lsh(corpus, num_vector, seed=143)
    
    for i, ix in enumerate(docs):
        start = time.time()
        result, num_candidates = query(corpus[ix,:], model, k=10, max_search_radius=3)
        end = time.time()
        
        query_time[num_vector].append(end-start)
        precision[num_vector].append(len(set(result['id']) & ground_truth[ix])/10.0)
        average_distance[num_vector].append(result['distance'][1:].mean())
        num_candidates_history[num_vector].append(num_candidates)
```

    num_vector = 5
    num_vector = 6
    num_vector = 7
    num_vector = 8
    num_vector = 9
    num_vector = 10
    num_vector = 11
    num_vector = 12
    num_vector = 13
    num_vector = 14
    num_vector = 15
    num_vector = 16
    num_vector = 17
    num_vector = 18
    num_vector = 19
    


```python
plt.figure(figsize=(7,4.5))
plt.plot(range(5,20), [np.mean(average_distance[i]) for i in xrange(5,20)], linewidth=4, label='Average over 10 neighbors')
plt.xlabel('# of random vectors')
plt.ylabel('Cosine distance')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size':16})
plt.tight_layout()

plt.figure(figsize=(7,4.5))
plt.plot(range(5,20), [np.mean(precision[i]) for i in xrange(5,20)], linewidth=4, label='Precison@10')
plt.xlabel('# of random vectors')
plt.ylabel('Precision')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size':16})
plt.tight_layout()

plt.figure(figsize=(7,4.5))
plt.plot(range(5,20), [np.mean(query_time[i]) for i in xrange(5,20)], linewidth=4, label='Query time (seconds)')
plt.xlabel('# of random vectors')
plt.ylabel('Query time (seconds)')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size':16})
plt.tight_layout()

plt.figure(figsize=(7,4.5))
plt.plot(range(5,20), [np.mean(num_candidates_history[i]) for i in xrange(5,20)], linewidth=4,
         label='# of documents searched')
plt.xlabel('# of random vectors')
plt.ylabel('# of documents searched')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size':16})
plt.tight_layout()
```


![png](output/output_234_0.png)



![png](output/output_234_1.png)



![png](output/output_234_2.png)



![png](output/output_234_3.png)


通过图表发现，随着随机向量数目的增加，搜索时间降低，因为每一个区域含有更少的数据点；但是近邻点的平均距离离query point更远。另一方面，当随机向量数目少的时候，结果更接近brute-force搜索：在一个区域中有很多店，所以对query point在的区域进行搜索得到很多点；这样包括邻近区域时，可能就是搜索几乎所有点，和brute-force搜索一样。
