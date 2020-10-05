# -*- coding: utf-8 -*-
# https://blog.csdn.net/littlely_ll/article/details/79082776
# Thanks to the orignial author.


from collections import defaultdict
import math

class MaxEnt(object):
    def __init__(self):
        self.feats = defaultdict(int)
        self.trainset = []  #训练集
        self.labels = set() #标签集

    def load_data(self,file):
        for line in open(file):
            fields = line.strip().split()
            if len(fields) < 2: continue    #特征数要大于等于2列
            label = fields[0]   #默认第一列为标签
            self.labels.add(label)
            for f in set(fields[1:]):
                self.feats[(label,f)] += 1    #(label,f)元组为特征
            self.trainset.append(fields)

    # Initialize parameter: 
    # 1. Compute empirical expectation of f_i, $\tilde{E}(f_i)$.
    # 2. Comput $C$ for GIS training. $C =max \sum_i f_i(x,y)$, for each pair of $(x,y)$ in training set
    # 3. Assign index for each feature function
    def __initparams(self):
        # Import train set.
        self.size = len(self.trainset)
        self.M = max([len(record)-1 for record in self.trainset])   #GIS训练算法的M参数 (or, $C$)
        self.ep_ = [0.0] * len(self.feats)
        for i,f in enumerate(self.feats):
            #计算经验分布的特征期望 $\tilde{E}(f_i)$
            self.ep_[i] = float(self.feats[f])/float(self.size)     
            self.feats[f] = i   #为每个特征函数分配id

        self.w = [0.0]*len(self.feats)  #初始化权重
        self.lastw = self.w

    #计算每个特征权重的指数 $\sum_{i=1}^m \lambda_i$ f_i(\vec{x},y), given fixed x and y.
    def probwgt(self, features, label):     
        wgt = 0.0
        for f in features:
            if (label,f) in self.feats:
                wgt += self.w[self.feats[(label,f)]]
#                print('self.w is:', self.w)
#                print('self.feats[(label,f) is:', self.feats[(label,f)]) #index of feature function
#                print('self.w[self.feats[(label,f)]] is:', self.w[self.feats[(label,f)]]) 
        return math.exp(wgt)

    """calculate feature expectation on model distribution
    """
    def calprob(self, features):    #计算条件概率
        wgts = [(self.probwgt(features,l),l) for l in self.labels]
        Z = sum([w for w,l in wgts])    #归一化参数
        prob = [(w/Z,l) for w,l in wgts]    #概率向量
#        print('wgts', wgts, 'Z', Z, 'prob',prob)
        return prob

    # Compute the expection of all f_i, $E(f_i)$. 
    def Ep(self):   #特征函数 $E(f_i)$
        ep = [0.0] * len(self.feats)
        for record in self.trainset:    #从训练集中迭代输出特征
            features = record[1:]
            prob = self.calprob(features)   #计算条件概率p(y|x)
            for f in features:
                for w,l in prob:
                    if (l,f) in self.feats: #来自训练数据的特征
                        idx = self.feats[(l,f)] #获取特征id
                        ep[idx] += w * (1.0/self.size)   # sum(1/N * f(y,x) * p(y|x)), p(x)=1/N
        return ep

    def __convergence(self,lastw,w):    #收敛条件
        for w1,w2 in zip(lastw,w):
            if abs(w1-w2) >= 0.01: return False
        return True

    def train(self,maxiter=1000):   #训练主函数，默认迭代次数1000
        self.__initparams() #初始化参数
#        for i in xrange(maxiter):
        for i in range(maxiter):
#            print "iter %d ..." % i+1
            print("iter {0}...".format(i+1))
            self.ep = self.Ep()     #计算模型分布的特征期望
            self.lastw = self.w[:]
            for i,win in enumerate(self.w):
                delta = 1.0/self.M * math.log(self.ep_[i]/self.ep[i])  #迭代公式！!!
#                delta = 1.0/self.M * math.log(self.ep[i]/self.ep_[i]) 
                self.w[i] += delta  #更新w
#            print self.w, self.feats
            print(self.w, self.feats)
            if self.__convergence(self.lastw,self.w):    #判断算法是否收敛
                break

    def predict(self,input):
        features = input.strip().split()
        prob = self.calprob(features)
        prob.sort(reverse=True)
        return prob

if __name__ == "__main__":
    model = MaxEnt()
    model.load_data('inputdata.txt')
    model.train()
    print("===================================\n")
    output = model.predict("rainy wet happy")
    print(output)
#    print model.predict("sunny happy")
