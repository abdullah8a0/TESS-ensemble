from lcobj import LC
import accuracy_model
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import CategoricalNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn import svm
from sklearn.model_selection import cross_val_score
from cluster_anomaly import cluster_and_plot,scale_simplify,tsne_plot
import run_classif


def classify(sector):
    data = accuracy_model.Data(sector,'scalar',partial=False)
    data_api = accuracy_model.Data(sector,'scalar')
    tags = data.stags

    passed_tags= cluster_and_plot(tags=tags,datafinder=data_api,verbose=True,suppress=True,score=run_classif.blob_score)
    if passed_tags.shape[0]>1000:
        passed_tags = cluster_and_plot(tags=passed_tags,datafinder=data_api,verbose=True,suppress=True,score=run_classif.struct_score)
    tags = passed_tags
    model = accuracy_model.AccuracyTest(tags)
    ind,tags = model.insert(99)
    data.new_insert(ind)
    sdata = data.get_some(tags,type='scalar')[:,(7,12,15,17,21,22,23,26,27)]
    normed_data = scale_simplify(sdata,False,9)

    labels = [True if i in ind else False for i in range(len(tags))]

    #clf = RandomForestClassifier(n_estimators=2,max_depth=2) 
    clf = svm.SVC()

    clf.fit(normed_data,labels)
    print('fitted')
    new_labels = clf.predict(normed_data).astype('int32')
    print('predicted')
    scores = cross_val_score(clf, normed_data, labels, cv=5)
    print(scores)
    tsne_plot(sector,tags,normed_data,new_labels,TOI=data)
    print(clf.decision_path(normed_data)[1])

if __name__ == '__main__':
    classify(32)