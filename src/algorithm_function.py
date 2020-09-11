import pandas as pd
import math
import numpy as np
import re
import random
from anytree import Node, RenderTree, PreOrderIter
from anytree.util import commonancestors

qiClass = {}
#Hàm phân loại quasi-identifiers theo numeric or categorical or hierarchy
def classifyNumCtg(df, qi):
    #Convert numeric to it's type
    for i in qi:
        df[i] = pd.to_numeric(df[i], errors='ignore')
    #Classify QI Numeric and QI Categorical
    qiNumeric = []
    qiCategorical = []
    for i in qi:
        if df[i].dtype == 'O':
            qiCategorical.append(i)
        else:
            qiNumeric.append(i)
    return {'numeric':qiNumeric, 'categorical':qiCategorical, 'hierarchy':{}}

#Hàm chuyển đổi input từ user sang dạng cây phân cấp:
#TREE STRING EXAMPLE: A(B/C) với A là root B, C là node con của A
#-------------------- A(B(D/E)/C(G/H)) A là root; D, E là node con của B
#-------------------- G, H là node con của C; B, C là node con của A 
def toTree(tree_string):
    x = re.split('[()]', tree_string)
    x = [i for i in x if i]
    rootNode = 0
    parentNode = []
    leafNode = []
    counterParent = 0
    counterLeaf = 0
    for item in x:
        if len(re.findall('/', item)) == 0:
            flag = 0
            if rootNode == 0:
                rootNode = Node(item)
            else:
                if counterParent == 0:
                    parentNode.append(Node(item, parent=rootNode))
                else:
                    parentNode.append(Node(item, parent=parentNode[counterParent-1]))
                counterParent = counterParent + 1
        else:
            if not re.search('^/', item):
                if flag != 1:
                    for i in re.split('/', item):
                        try:
                            leafNode.append(Node(i, parent=parentNode[counterParent-1]))
                        except:
                            leafNode.append(Node(i, parent=rootNode))
                        counterLeaf = counterLeaf + 1
                if flag == 1:
                    counterLeaf = counterLeaf - 1
                    parentNode.append(leafNode[counterLeaf])
                    leafNode.remove(leafNode[counterLeaf])
                    counterParent = counterParent + 1
                    for i in re.split('/', item):
                        leafNode.append(Node(i, parent=parentNode[counterParent-1]))
                        counterLeaf = counterLeaf + 1
                flag = 1
            else:
                if len(re.findall('/', item)) < 2:
                    if flag != 2:
                        if item == x[-1]:
                            if str('))' + item) in tree_string:
                                leafNode.append(Node(item.split('/')[1], parent=parentNode[counterParent-1].parent.parent))
                            else:
                                leafNode.append(Node(item.split('/')[1], parent=parentNode[counterParent-1].parent))
                            counterLeaf = counterLeaf + 1
                        else:
                            if str('))))' + item) in tree_string:
                                parentNode.append(Node(item.split('/')[1], parent=rootNode))
                            elif str('))' + item) in tree_string:
                                parentNode.append(Node(item.split('/')[1], parent=parentNode[counterParent-1].parent.parent))
                            else:
                                parentNode.append(Node(item.split('/')[1], parent=parentNode[counterParent-1].parent))
                            counterParent = counterParent + 1
                    if flag == 2:
                        counterParent = counterParent - 1
                        leafNode.append(parentNode[counterParent])
                        parentNode.remove(parentNode[counterParent])
                        counterLeaf = counterLeaf + 1
                        if str(')' + item) in tree_string:
                            leafNode.append(Node(item.split('/')[1], parent=parentNode[counterParent-1].parent.parent.parent))
                        else:
                            leafNode.append(Node(item.split('/')[1], parent=parentNode[counterParent-1].parent.parent))
                        counterLeaf = counterLeaf + 1
                    flag = 2
                else:
                    pass
    return [rootNode, parentNode, leafNode]

#Hàm chuyển QI kiểu categorical về dạng numeric:
def convertQIValue(df, qilist):
    fullMapDict = {}
    for qi in qilist:
        oldValue = df.sort_values(by=qi)[qi].unique()
        newValue = []
        for i in range(len(df[qi].unique())):
            if len(df[qi].unique()) == 2:
                #newValue.append(qi + str(i))
                newValue.append(i)
            else:
                newValue.append(i+1)
        mapDict = dict(zip(oldValue, newValue))
        df[qi] = df[qi].replace(mapDict)
        fullMapDict.update({qi: mapDict})
    return fullMapDict

## Hàm tính khoảng cách giữa 2 records (rows)
NUM_CATEGORICAL = True
def distance(r1, r2, df, qiClass): #(row1, row2, dataframe, quasi-identifers class)
    totalDistance = 0
    numDistance = 0
    catgDistance = 0
    hieDistance = 0
    for qi in qiClass['numeric']:
        if df[qi].max() == df[qi].min():
            numDistance += 0
        else:
            numDistance += abs(r1[qi]-r2[qi])/(df[qi].max() - df[qi].min())
    for qi in qiClass['categorical']:
        if NUM_CATEGORICAL:
            if df[qi].max() == df[qi].min():
                catgDistance += 0
            else:
                catgDistance += abs(r1[qi]-r2[qi])/(df[qi].max() - df[qi].min())
        else:
            if r1[qi] != r2[qi]:
                catgDistance += 1
    for qi in qiClass['hierarchy'].keys():
        if r1[qi] != r2[qi]:
            r1_ac = set(r1[qi].ancestors)
            r2_ac = set(r2[qi].ancestors)
            try:
                comanc = list(r1_ac.intersection(r2_ac))[-1]
            except:
                comanc = qiClass['hierarchy'][qi]['root']
            numLeavesCA = len(list(PreOrderIter(comanc, filter_=lambda node: node.is_leaf)))
            totalLeaves = len(qiClass['hierarchy'][qi]['children'])
            hieDistance += numLeavesCA/totalLeaves
    totalDistance = numDistance + catgDistance + hieDistance
    return totalDistance

#Convert string type to tree node type:
#Hàm chuyển giá trị kiểu chuỗi sang kiểu node cây
def toNode(df, qiClass):
    for qi in qiClass['hierarchy']:
        leavesDict = {}
        for leaf in qiClass['hierarchy'][qi]['children']:
            leavesDict.update({leaf.name:leaf})
        df[qi] = df[qi].replace(leavesDict)
    return df

## Định nghĩa lớp phân cụm:
dfsorted = pd.DataFrame()
class Cluster:
    def __init__(self, centroid):
        self.centroid = centroid
        self.records = pd.DataFrame(columns=self.centroid.index)
        self.records = self.records.append(self.centroid)
        self.informationLoss = 0
    def updateCentroid(self, qiClass):
        for qi in qiClass['numeric']:
            self.centroid[qi] = self.records[qi].mean()
        for qi in qiClass['categorical']:
            if NUM_CATEGORICAL:
                self.centroid[qi] = self.records[qi].mean()
            else:
                if len(self.records[qi].unique()) > 1:
                    self.centroid[qi] = qi
                else:
                    self.centroid[qi] = self.records[qi].unique()[0]
        for qi in qiClass['hierarchy']:
            listAncestor = []
            for item in self.records[qi].unique():
                listAncestor.append(set(item.ancestors))
            for n in range(len(listAncestor)):
                commonA = listAncestor[0].intersection(listAncestor[n])
            commonA = list(commonA)
            lowestCommonA = commonA[0]
            for i in commonA:
                if len(i.path) > len(lowestCommonA.path):
                    lowestCommonA = i
            self.centroid[qi] = lowestCommonA
    def updateIL(self, qiClass, dfsorted):
        PIL = 0
        numDistance = 0
        catgDistance = 0
        hieDistance = 0
        for qi in qiClass['numeric']:
            if dfsorted[qi].max() == dfsorted[qi].min():
                numDistance += 0
            else:
                numDistance += abs(self.records[qi].max() - self.records[qi].min())/(dfsorted[qi].max() - dfsorted[qi].min())
        for qi in qiClass['categorical']:
            if NUM_CATEGORICAL:
                if dfsorted[qi].max() == dfsorted[qi].min():
                    catgDistance += 0
                else:
                    catgDistance += abs(self.records[qi].max() - self.records[qi].min())/(dfsorted[qi].max() - dfsorted[qi].min())
            else:
                if len(self.records[qi].unique()) > 1:
                    catgDistance += 1
        for qi in qiClass['hierarchy'].keys():
            listAncestor = []
            for item in self.records[qi].unique():
                listAncestor.append(set(item.ancestors))
            for n in range(len(listAncestor)):
                commonA = listAncestor[0].intersection(listAncestor[n])
            lowestCommonA = list(commonA)[-1]

            numLeavesCA = len(list(PreOrderIter(lowestCommonA, filter_=lambda node: node.is_leaf)))
            totalLeaves = len(qiClass['hierarchy'][qi]['children'])
            hieDistance += numLeavesCA/totalLeaves
        PIL = numDistance + catgDistance + hieDistance
        self.informationLoss = PIL

## Hàm chạy giải thuật ##
def anonymizeDataFrame(dfin, LIST_QI, NUM_CATEGORICAL=True, k=2):
    #Phân loại Quasi-identifiers
    qiClass = classifyNumCtg(dfin, LIST_QI)

    #Chuyển các categorical QIs về dạng số nếu NUM_CATEGORICAL = True
    if NUM_CATEGORICAL:
        conversion = convertQIValue(dfin, qiClass['categorical'])
    else:
        conversion = {}

    #Sort the table by QIs
    dfsorted = dfin.sort_values(by=LIST_QI)

    #Get K number of clusters
    K = dfsorted.shape[0]//int(k) #Chia lấy phần nguyên dưới
    
    #Random select K distinct records, save to df P:
    P = dfsorted.drop_duplicates().sample(n=K)
    P = toNode(P, qiClass)

    #Table of records without K selected records
    T = dfsorted[~dfsorted.isin(P)].dropna().sort_values(by=LIST_QI)
    T = toNode(T, qiClass)
    T_temp = toNode(dfsorted.copy(), qiClass)
    dfsorted = toNode(dfsorted, qiClass)

    #Khởi tạo cụm
    clusters = list()
    for index, row in P.iterrows():
        clusters.append(Cluster(row))

    #Phân cụm
    while not T.empty:
        #Calculate the distance between each row from T to each cluster's centroid
        distancesList = []
        for cluster in clusters:
            distancesList.append(distance(T.iloc[0], cluster.centroid, dfsorted, qiClass))
    
        #Get the min distance
        minDistance = min(distancesList)
        
        #Add row to its closest cluster and update centroid.
        clusters[distancesList.index(minDistance)].records = clusters[distancesList.index(minDistance)].records.append(T.iloc[0])
        clusters[distancesList.index(minDistance)].updateCentroid(qiClass)
        
        #Drop row added from T
        T.drop(index=T.iloc[0].name, inplace=True)
        
    #Điều chỉnh cụm
    for cluster in clusters:
        cluster.records = T
    for i in range(len(clusters)):
        newColString = 'distance_to_cluster_' + str(i)
        T_temp[newColString] = T_temp.apply(lambda x: distance(x, clusters[i].centroid, dfsorted, qiClass), axis=1)
    
    #Get random clusters
    randomClusterList = []
    while len(randomClusterList) < K:
        r = random.randint(0, K-1)
        if r not in randomClusterList:
            randomClusterList.append(r)
    
    while not T_temp.empty:
        if randomClusterList != []:
            for i in randomClusterList:
                colString = 'distance_to_cluster_' + str(i)
                T_temp = T_temp.sort_values(by=colString, ascending=True)
                clusters[i].records = clusters[i].records.append(T_temp.iloc[0:k])
                T_temp = T_temp.iloc[k:]
                if clusters[i].records.shape[0] == k:
                    randomClusterList.remove(i)
        else:
            indexCount = 0
            tempIL = 0
            maxIL = 0
            for i in range(len(clusters)):
                clusters[i].updateIL(qiClass, dfsorted)
                tempIL = clusters[i].informationLoss
                if tempIL > maxIL:
                    maxIL = tempIL
                    indexCount = i
            clusters[indexCount].records = clusters[indexCount].records.append(T_temp)
            T_temp = T
            if T_temp.empty:
                break
    
    #Chuyển dữ liệu số lại như cũ
    inv_conversion = {}
    for qi in conversion:
        inv_conversion.update({qi: {v: k for k, v in conversion[qi].items()}})
    for cluster in clusters:
        for qi in inv_conversion:
            cluster.records[qi] = cluster.records[qi].replace(inv_conversion[qi])

    #Remap Value-Label for anonymity
    for cluster in clusters:
        for qi in qiClass['numeric']:
            oldValue = cluster.records[qi].unique()
            newValue = []
            for i in range(len(oldValue)):
                if cluster.records[qi].min() == cluster.records[qi].max():
                    newValue.append(str(cluster.records[qi].min()))
                else:
                    newValue.append('[' + str(cluster.records[qi].min()) + '-' + str(cluster.records[qi].max()) + ']')
            mapDict = dict(zip(oldValue, newValue))
            cluster.records[qi] = cluster.records[qi].replace(mapDict)
        for qi in qiClass['categorical']:
            if NUM_CATEGORICAL:
                oldValue = cluster.records[qi].unique()
                newValue = []
                newValueString = ''
                for i in oldValue:
                    newValueString += '|' + str(i) + '|'
                for i in range(len(oldValue)):
                    newValue.append(newValueString)
                mapDict = dict(zip(oldValue, newValue))
                cluster.records[qi] = cluster.records[qi].replace(mapDict)
            else:
                oldValue = cluster.records[qi].unique()
                newValue = []
                for i in range(len(oldValue)):
                    newValue.append(cluster.centroid[qi])
                mapDict = dict(zip(oldValue, newValue))
                cluster.records[qi] = cluster.records[qi].replace(mapDict)
        for qi in qiClass['hierarchy']:
            oldValue = cluster.records[qi].unique()
            newValue = []
            for i in range(len(oldValue)):
                newValue.append(cluster.centroid[qi].name)
            mapDict = dict(zip(oldValue, newValue))
            cluster.records[qi] = cluster.records[qi].replace(mapDict)
    
    dfout = T
    for cluster in clusters:
        dfout = pd.concat([dfout, cluster.records], ignore_index=True)
    for i in range(len(clusters)):
        dfout.drop(columns='distance_to_cluster_'+str(i), inplace=True)

    return dfout