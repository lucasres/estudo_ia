from math import sqrt
from scipy.spatial import distance

class KNN():
    """
    This is my first classifier :)
    """
    def fit(self,data,labels):
        self.data = data
        self.labels = labels
        
        return self
    
    def predict(self,x_test):
        predicts = []
        for row in x_test:
            label = self.closest(row)
            predicts.append(label)
        return predicts
    
    def euc(self,a,b):
        #rs = sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
        return distance.euclidean(a,b)
            
    
    def closest(self,row):
        #initial best distance is alwas first
        best_distance = self.euc(row,self.data[0])
        best_index = 0
        #intering in data
        for i in range(1,len(self.data)):
            dist = self.euc(row,self.data[i])
            if dist < best_distance:
                best_distance = dist
                best_index = i
            print(dist,best_distance, best_index)
        return self.labels[best_index]

#enuns
enuns = {
    1:'Carro',
    0:'Moto',
}

#data

features = [
    [110,2],
    [125,2],
    [100,2],
    [110,2],
    [300,4],
    [278,4],
    [290,4],
    [260,4],
]
label = [
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
]

#make the aim
prop_x = float(input('X:'))
prop_y = float(input('Y:'))

#instancie classifier
clf = KNN()
#treaning
clf = clf.fit(features,label)
#predict
rs = clf.predict([[prop_x,prop_y]])
#print result
print(enuns[rs[0]])

