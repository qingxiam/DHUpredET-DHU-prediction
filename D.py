df1 = pd.read_csv('/content/drive/MyDrive/phage virion proteins/train_pseip.csv')
df2 = pd.read_csv('/content/drive/MyDrive/phage virion proteins/train_lsa.csv')
df3 = pd.read_csv('/content/drive/MyDrive/phage virion proteins/train_fst.csv')
df4 = pd.read_csv('/content/drive/MyDrive/phage virion proteins/train_d2v.csv')
df5 = pd.read_csv('/content/drive/MyDrive/phage virion proteins/train_bert.csv')



df6 = pd.read_csv('/content/drive/MyDrive/phage virion proteins/Dataset of DHU/dpcp pos train.csv')
df7 = pd.read_csv('/content/drive/MyDrive/phage virion proteins/Dataset of DHU/z curve pos train.csv')
df8 = pd.read_csv('/content/drive/MyDrive/phage virion proteins/Dataset of DHU/RNA-binary train-pos.csv')
df9 = pd.read_csv('/content/drive/MyDrive/phage virion proteins/Dataset of DHU/ps2 pos train.csv')

tdf6 = pd.read_csv('/content/drive/MyDrive/phage virion proteins/Dataset of DHU/dpcp pos test.csv')
tdf7 = pd.read_csv('/content/drive/MyDrive/phage virion proteins/Dataset of DHU/z curve pos test.csv')
tdf8 = pd.read_csv('/content/drive/MyDrive/phage virion proteins/Dataset of DHU/Rna binary pos test.csv')
tdf9 = pd.read_csv('/content/drive/MyDrive/phage virion proteins/Dataset of DHU/ps2 pos test.csv')
tdf4 = pd.read_csv('/content/drive/MyDrive/phage virion proteins/test_d2v.csv')
tdf2 = pd.read_csv('/content/drive/MyDrive/phage virion proteins/test_lsa.csv')
tdf3 = pd.read_csv('/content/drive/MyDrive/phage virion proteins/test_fst.csv')
tdf1 = pd.read_csv('/content/drive/MyDrive/phage virion proteins/test_pseip.csv')
tdf5 = pd.read_csv('/content/drive/MyDrive/phage virion proteins/test_bert.csv')

df10 = pd.read_csv('/content/drive/MyDrive/phage virion proteins/Dataset of DHU/train_physicio-merge.csv')
tdf10 = pd.read_csv('/content/drive/MyDrive/phage virion proteins/Dataset of DHU/test_psyco.csv')

df11 = pd.read_csv('/content/drive/MyDrive/phage virion proteins/Dataset of DHU/ZPB merge train.csv')
tdf11 = pd.read_csv('/content/drive/MyDrive/phage virion proteins/Dataset of DHU/ZPB merge test.csv')

df12 = pd.read_csv('/content/drive/MyDrive/phage virion proteins/Dataset of DHU/ALl train.csv')
tdf12 = pd.read_csv('/content/drive/MyDrive/phage virion proteins/Dataset of DHU/ALL test.csv')

X_train9 = df9.drop('Target', axis=1)
y_train9 = df9['Target']
X_test9 = tdf9.drop('Target', axis=1)
y_test9 = tdf9['Target']

from sklearn.semi_supervised import LabelPropagation
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import ElasticNet
#from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.datasets import make_classification

kernel = 1.0 * RBF(length_scale=1.0)
gpc = GaussianProcessClassifier(kernel=kernel)

tree_clf1 = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_clf2 = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_clf3 = DecisionTreeClassifier(max_depth=7, random_state=42)

# Create an ensemble (Decision Jungle) using voting
dj = VotingClassifier(estimators=[('dt1', tree_clf1), ('dt2', tree_clf2), ('dt3', tree_clf3)], voting='hard')


lr= LogisticRegression(random_state=10,penalty='l2',solver='lbfgs',multi_class="ovr")
rf= RandomForestClassifier(n_estimators=100,criterion='entropy',max_features="sqrt",random_state=100)
svc = SVC(kernel='poly', degree = 3)
#svm = svm.SVC(kernel='linear', C=1)
ridge=RidgeClassifier()
ada=AdaBoostClassifier()
sgd=SGDClassifier()
ex=ExtraTreesClassifier(n_estimators=100, random_state=10, max_depth=None, min_samples_split=4, min_samples_leaf=1, max_features='auto', bootstrap=False, class_weight=None, criterion='gini', n_jobs=None)
la=LinearDiscriminantAnalysis()
lsv=LinearSVC()
dt=DecisionTreeClassifier(max_depth=10, criterion='entropy', min_samples_split=10, min_samples_leaf=1, max_features=15 , random_state=42)
mlp= MLPClassifier(activation='relu', solver='lbfgs', max_iter=160, random_state=100,alpha=0.0001,learning_rate='invscaling')
knn= KNeighborsClassifier(n_neighbors=100,metric='manhattan',weights='distance',algorithm="kd_tree")
lgbm = LGBMClassifier(num_leaves=10, max_depth=10,n_estimators=1000,colsample_bytree=0.8,min_child_samples=3)
xgb = XGBClassifier(n_estimators=100, max_depth=100, learning_rate=0.1, subsample=1.0, colsample_bytree=1.0, reg_alpha=30, reg_lambda=30, gamma=0, min_child_weight=1)
#cat=CatBoostClassifier(iterations=100, depth=10, learning_rate=0.1, loss_function='Logloss', random_seed=42, l2_leaf_reg=3,  bagging_temperature=1, scale_pos_weight=1)
lp = LabelPropagation(kernel='knn',max_iter=1000)
iso_reg = IsotonicRegression()
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
qda = QuadraticDiscriminantAnalysis()
#VOTING (HARD)
from sklearn.ensemble import  VotingClassifier
vth1 = VotingClassifier(estimators=[('lr', lr), ('knn', knn), ('dt', dt),('mlp',mlp)], voting='soft')
vth2 = VotingClassifier(estimators=[('xgb', xgb), ('lr', lr), ('knn', knn), ('dt', dt)], voting='hard')
vth3 = VotingClassifier(estimators=[('xgb', xgb), ('lr', lr), ('knn', knn),('lgbm',lgbm)], voting='hard')
vth4 = VotingClassifier(estimators=[ ('lgbm', lgbm),('rf',rf)], voting='hard')

vth5 = VotingClassifier(estimators=[ ('lgbm', lgbm),('knn',knn)], voting='hard')
vth6 = VotingClassifier(estimators=[ ('lgbm', lgbm),('dt',dt)], voting='hard')
vth7 = VotingClassifier(estimators=[ ('lgbm', lgbm),('mlp',mlp)], voting='hard')
vth8 = VotingClassifier(estimators=[ ('dt', dt),('xgb', xgb)], voting='hard')
vth9 = VotingClassifier(estimators=[ ('lgbm', lgbm), ('rf', rf), ('ex', ex)], voting='hard')

#STACK
from sklearn.ensemble import StackingClassifier
stk1 = StackingClassifier(estimators=[('lgbm', lgbm),('xgb',xgb)], final_estimator=LogisticRegression())
stk2 = StackingClassifier(estimators=[('dt',dt),('knn', knn)], final_estimator=LogisticRegression())
stk3 = StackingClassifier(estimators=[('xgb',rf),('knn', knn)], final_estimator=LogisticRegression())
stk4 = StackingClassifier(estimators=[('xgb', xgb), ('lr', lr), ('knn', knn),('lgbm',lgbm)], final_estimator=LogisticRegression())
stk5 = StackingClassifier(estimators=[('xgb', xgb), ('mlp', mlp), ('dt', dt),('lgbm',lgbm)], final_estimator=LogisticRegression())
stk6 = StackingClassifier(estimators=[('xgb', xgb), ('rf', rf), ('ex', ex)], final_estimator=LogisticRegression())

model_list = [lr,rf,svc,ada,ex,dt,mlp,knn,lgbm,xgb,gpc,la,qda,dj,lp,stk1,stk2,stk3,stk4,stk6,vth1,vth2,vth3,vth4,vth5,vth6,vth7,vth8,vth9]

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score


# Instantiate the Extra Trees Classifier model
extra_trees_model = ExtraTreesClassifier(n_estimators=100, random_state=10, max_depth=None, min_samples_split=4, min_samples_leaf=1, max_features='auto', bootstrap=False, class_weight=None, criterion='gini', n_jobs=None)

# Train the model
extra_trees_model.fit(X_train9, y_train9)

# Make predictions on the test set
y_pred9 = extra_trees_model.predict(X_test9)

# Calculate accuracy
accuracy = accuracy_score(y_test9, y_pred9)
print("Accuracy:", accuracy)

/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.
  warn(
Accuracy: 0.8524590163934426