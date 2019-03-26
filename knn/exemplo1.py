from sklearn import tree

#enuns
enuns = {
    0:'Carro',
    1:'Moto',
}

#data

features = [
    [110,2],
    [125,2],
    [200,4],
    [150,4],
]
label = [
    0,
    0,
    1,
    1,
]

#make the aim
prop_x = input('X:')
prop_y = input('Y:')

#instancie classifier
clf = tree.DecisionTreeClassifier()
#treaning
clf = clf.fit(features,label)
#predict
rs = clf.predict([[prop_x,prop_y]])
#print result
print(enuns[rs[0]])

