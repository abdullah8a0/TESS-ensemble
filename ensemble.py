from lcobj import LC
import accuracy_model
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score
import cluster_anomaly


def classify(sector):
    data = accuracy_model.Data(sector,'scalar')
    tags = data.stags
    model = accuracy_model.AccuracyTest(tags)
    ind,tags = model.insert(99)
    data.new_insert(ind)
    sdata = data.get_all(type='scalar')
    normed_data = cluster_anomaly.scale_simplify(sdata,False,16)

    labels = [True if i in ind else False for i in range(len(tags))]

    #clf = AdaBoostClassifier(n_estimators=100) 
    clf = RandomForestClassifier(max_depth=2) 
    #clf = svm.SVC()

    clf.fit(normed_data,labels)
    print('fitted')
    new_labels = clf.predict(normed_data)
    print('predicted')
    scores = cross_val_score(clf, normed_data, labels, cv=5)
    print(scores)
    cluster_anomaly.tsne_plot(sector,tags,normed_data,new_labels,TOI=data)

if __name__ == '__main__':
    classify(38)