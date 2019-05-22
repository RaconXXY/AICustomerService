#!/usr/bin/python
# -*- coding: utf-8 -*-

from data_prepare import *

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from train_model import *

def star_process(X_train,y_train,X_test,y_test):
    # Step2 分类模型
    num_folds = 5


    # Step3 定义交叉验证，及模型gbm参数
    rand_seed = 456
    kfold = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=rand_seed
    )
    lgb_param = {
        'objective':'binary',
        'metric':'binary_logloss',
        'boosting':'gbdt',
        'device':'cpu',
        'feature_fraction': 1,     #抽取所有特征的0.75个进行训练
        'num_leaves':16,
        'learning_rate':0.01,
        'verbose':1,
        'bagging_seed':rand_seed,
        'feature_fraction_seed':rand_seed
    }
    y_test_pred = np.zeros((len(X_test),5))

    for fold_num,(ix_train,ix_val) in enumerate(kfold.split(X=X_train,y=y_train)):
        # 准备数据
        X_fold_train = X_train[ix_train]
        X_fold_val = X_train[ix_val]

        y_fold_train = y_train[ix_train]
        y_fold_val = y_train[ix_val]
        print ('train fold {} of {} ......'.format((fold_num + 1), 5))
        # 定义模型
        lgb_data_train = lgb.Dataset(X_fold_train,y_fold_train)
        lgb_data_val = lgb.Dataset(X_fold_val,y_fold_val)
        evals_res = {}

        model = lgb.train(
            params=lgb_param,
            train_set= lgb_data_train,
            valid_sets=[lgb_data_train,lgb_data_val], # 训练集和测试集都需要验证
            valid_names = ['train','val'],
            evals_result= evals_res,
            num_boost_round=2500,
            early_stopping_rounds=10,
            verbose_eval=False,
        )
        fold_train_score = evals_res['train'][lgb_param['metric']]
        fold_val_score = evals_res['val'][lgb_param['metric']]

        print ('fold {}: {} rounds ,train loss {:.6f}, val loss {:.6f}'.format(
            (fold_num+1),
            len(fold_train_score),
            fold_train_score[-1],
            fold_val_score[-1]
        ))

        y_test_pred[:,fold_num] = model.predict(X_test).reshape(-1)

    print (y_test_pred.shape, '0')
    y_test_p = np.mean(y_test_pred,axis=1)

    # np.save('y_test_pre','pre.npy')
    print (y_test_pred.shape ,'1')
    print (y_test_p.shape ,'2')

    for index,pre in enumerate(y_test_p):
        if pre >=0.5:
            y_test_p[index] = 1
        else:
            y_test_p[index] = 0

    print (y_test.shape,'3')
    print (accuracy_score(y_test,y_test_p))
    print (f1_score(y_test,y_test_p))

if __name__ == '__main__':
    # step1 选出的特征
    feature_names_list = [
        'dl_siamese_lstm_manDist',
        'dl_siamese_lstm_attention',
        'dl_siamese_lstm_dssm',
        'nlp_sentece_length_diff',
        'nlp_edit_distance',
        'nlp_ngram',
        'nlp_sentece_diff_some',
        'nlp_doubt_sim',
        'nlp_word_embedding_sim'
    ]
    # 加载数据
    df_train,df_test,feature_index_ix = project.load_feature_lists(feature_names_list)
    # 查看抽取的特征情况
    feature_view_df = pd.DataFrame(feature_index_ix, columns=['feature_name', 'start_index', 'end_index'])
    print (feature_view_df)

    print (df_train.head(20))
    print (df_train.tail(20))
    y_train = np.array(project.load(project.features_dir + 'y_0.6_train.pickle'))

    y_test = pd.read_csv(project.data_dir + 'atec_nlp_sim_test_0.4.csv', sep='\t', header=None,
                                names=["index", "s1", "s2", "label"])['label'].values.reshape((-1))

    X_test = df_test.values
    X_train = df_train.values

    gnb_cls = GussianNBClassifier()
    gnb_oop_train,gnb_oofp_val = gnb_cls.get_model_out(X_train,y_train,X_test)
    print (gnb_oofp_val[0:25])

    rf_cls = RFClassifer()
    rf_oop_train, rf_oofp_val = rf_cls.get_model_out(X_train, y_train, X_test)
    print (rf_oofp_val[0:25])

    lg_cls = LogisicClassifier()
    lg_oop_train, lg_oofp_val = lg_cls.get_model_out(X_train, y_train, X_test)
    print (lg_oofp_val[0:25])

    dt_cls = DecisionClassifier()
    dt_oop_train, dt_oofp_val = dt_cls.get_model_out(X_train, y_train, X_test)
    print (dt_oofp_val[0:25])


    # 构造输入
    input_train = [gnb_oop_train,rf_oop_train,lg_oop_train,dt_oop_train]
    input_test = [gnb_oofp_val,rf_oofp_val,lg_oofp_val,dt_oofp_val]

    stacked_train = np.concatenate([data.reshape(-1,1) for data in input_train],axis=1)
    stacked_test = np.concatenate([data.reshape(-1,1) for data in input_test],axis=1)

    # stacking 第二层模型训练

    second_model = DecisionTreeClassifier(max_depth=3,class_weight={0: 1, 1: 4})
    second_model.fit(stacked_train,y_train)

    y_test_p = second_model.predict(stacked_test)

    for index,pre in enumerate(y_test_p):
        if pre >=0.5:
            y_test_p[index] = 1
        else:
            y_test_p[index] = 0

    print ("recall_score: ", recall_score(y_test,y_test_p))
    print ("accuracy_score: ", accuracy_score(y_test,y_test_p))
    print ("F1: ", f1_score(y_test,y_test_p))
