# nlp
## naive bayes and sentiment classification
> 
$c\hat=argmaxP(c|d)$ posterior probability
> $P(x|y)=\frac{P(y|x)*P(x)}{P(y)}$
so
$$\hat c  =argmaxP(c|d)(c\isin C )=\frac{P(d|c)*P(c)}{P(d)\tag{1}}$$
当前分类的概率就是 已知分类是此文档的概率*是此分类的概率（此分类在所有分类中的占比） 
由于分类的结果就是在所有预测中找到一个最大值，而分母P(d)对于所有分类都是一样的所以可以忽略  
在(1)中 $P(d|c)$是似然（likelihood）,$P(c)$是先验概率（prior）
在loss generalization 之前，可以把d（document）用一系列特征（feature）表示成
$$P(d|c)=P(f1f2f3...fk|c)\tag{2}$$
由于等式右侧的概率还是非常难以计算：已知分类计算一系列特征出现的概率  
因此接下来对于词分类主要有两个步骤的猜想：
- bagofwords
- naive bayes assumption
### bag of words
主要作用：
- 打乱次序 使得词义与顺序无关

## naive bayes assumption
- 将一系列特征的总体概率转化为每一个相互无关的（基于 bag of words）特征的概率乘积
最后的等式就是
$$c_{NB}=argmax_{c\isin C}P(c) multiple P(w_i|c)$$
取对数处理之后可以写成加法形式

一些优化方法
### binary multinomial naive Bayes 
原理：在分类的时候，一个词语的出现与否比它的出现次数重要，防止出现次数在一句话中的过多出现对结果的影响（十句话中，某个词在**每一句话中都出现了**比**只在一句话中出现十次**中出现更能代表这一类的特侦和）  
做法：删除一句话中重复出现的词语
### deal with negation
原理：否定词会对句子的含义产生影响，即使其他词是相同的
做法：将这一句话的剩余词语加上NOT_为前缀（作为新的词统计）
### use lexicons
