"""Обучение Product Matching v3. на ручной разметке из PostgreSQL.
Запуск: python train_product_matching_v3.py
product_matching_features_v3.py должен лежать в одной папке. Таблицы БД не изменяются.
"""
import json, os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from product_matching_features_v3 import PairFeatureExtractorV3,clean,same_conflict

BASE=Path(__file__).resolve().parent
load_dotenv(BASE/'.env')
OUT=BASE/'matching_results_v3'
LABELS=['same','alternative','different']
COLS=['name_a','name_b','brand_a','brand_b','category_a','category_b','quantity_a','quantity_b','unit_a','unit_b']
SQL='''SELECT a.product_id_a,a.product_id_b,a.human_label,
 COALESCE(NULLIF(p.name_a,''),NULLIF(x.original_name,''),x.name_clean,'') AS name_a,
 COALESCE(NULLIF(p.name_b,''),NULLIF(y.original_name,''),y.name_clean,'') AS name_b,
 x.brand AS brand_a,y.brand AS brand_b,x.category AS category_a,y.category AS category_b,
 x.total_quantity AS quantity_a,y.total_quantity AS quantity_b,
 x.normalized_unit AS unit_a,y.normalized_unit AS unit_b
 FROM grocery_clean.product_pair_annotations a
 LEFT JOIN grocery_clean.product_pair_candidates p ON p.product_id_a=a.product_id_a AND p.product_id_b=a.product_id_b
 LEFT JOIN grocery_clean.products_clean x ON x.product_id=a.product_id_a
 LEFT JOIN grocery_clean.products_clean y ON y.product_id=a.product_id_b'''

def groups_for(df):
    parent={}
    def root(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=root(parent[x])
        return parent[x]
    for a,b in zip(df.product_id_a,df.product_id_b):
        a,b=root(int(a)),root(int(b))
        if a!=b:parent[b]=a
    return np.array([root(int(a)) for a in df.product_id_a])

def group_split(df,test_size,seed):
    groups=groups_for(df)
    best=None
    for tr,te in GroupShuffleSplit(n_splits=400,test_size=test_size,random_state=seed).split(df,df.human_label,groups):
        train,test=df.iloc[tr],df.iloc[te]
        if min(train.human_label.value_counts().reindex(LABELS,fill_value=0))<2:continue
        if min(test.human_label.value_counts().reindex(LABELS,fill_value=0))<2:continue
        score=abs(len(test)/len(df)-test_size)+sum(abs(test.human_label.eq(k).mean()-df.human_label.eq(k).mean()) for k in LABELS)
        if best is None or score<best[0]:best=(score,tr,te)
    if best is None:raise ValueError('Не удалось получить независимое групповое разбиение с 3 классами.')
    return df.iloc[best[1]].copy(),df.iloc[best[2]].copy()

def predict_with_gate(pipe,df,threshold):
    probs=pipe.predict_proba(df[COLS]); classes=list(pipe.named_steps['model'].classes_)
    raw=pipe.predict(df[COLS]); final=[]; flagged=[]
    for i,(_,row) in enumerate(df.iterrows()):
        conflict=same_conflict(row)
        same_prob=float(probs[i,classes.index('same')])
        if raw[i]=='same' and (same_prob<threshold or conflict):
            # Not safe to auto-confirm; route to review, NOT relabel as alternative.
            final.append('review');flagged.append(True)
        else:final.append(raw[i]);flagged.append(False)
    return np.asarray(final),np.asarray(flagged),probs,classes

def main():
    cfg=dict(host=os.getenv('DB_HOST','localhost'),port=int(os.getenv('DB_PORT','5432')),
             dbname=os.getenv('DB_NAME','grocery_store'),user=os.getenv('DB_USER','postgres'))
    if os.getenv('DB_PASSWORD'):cfg['password']=os.getenv('DB_PASSWORD')
    with psycopg.connect(**cfg) as conn:
        with conn.cursor() as cur:
            cur.execute(SQL);df=pd.DataFrame(cur.fetchall(),columns=[c.name for c in cur.description])
    df=df[df.human_label.isin(LABELS)].copy()
    df=df[df.name_a.map(clean).ne('')&df.name_b.map(clean).ne('')].copy()
    if df.empty:raise ValueError('Нет размеченных пар.')
    if df.duplicated(['product_id_a','product_id_b']).any():raise ValueError('Дубликаты пар.')
    trainval,test=group_split(df,.2,42)
    train,val=group_split(trainval,.25,43)
    overlap=lambda a,b:(set(a.product_id_a)|set(a.product_id_b))&(set(b.product_id_a)|set(b.product_id_b))
    if overlap(train,val) or overlap(train,test) or overlap(val,test):raise RuntimeError('Пересечение товаров между выборками')
    OUT.mkdir(exist_ok=True)
    for name,part in [('train',train),('validation',val),('test',test)]:part.to_csv(OUT/f'{name}_pairs.csv',index=False,encoding='utf-8-sig')
    estimators={'logistic_regression':LogisticRegression(max_iter=3000,class_weight='balanced',random_state=42),
                'random_forest':RandomForestClassifier(n_estimators=350,min_samples_leaf=2,class_weight='balanced_subsample',random_state=42,n_jobs=-1)}
    results={}; fitted={}
    for name,est in estimators.items():
        pipe=Pipeline([('features',PairFeatureExtractorV3()),('model',est)])
        pipe.fit(train[COLS],train.human_label)
        # Threshold selected ONLY on validation. Test remains untouched.
        candidates=[]
        for t in [0.5,0.6,0.7,0.8,0.9]:
            pred,_,_,_=predict_with_gate(pipe,val,t)
            accepted=pred=='same'; tp=int(((val.human_label.to_numpy()=='same')&accepted).sum())
            precision=tp/max(1,int(accepted.sum()))
            candidates.append((precision,tp,t,int(accepted.sum())))
        viable=[c for c in candidates if c[1]>=3]
        chosen=max(viable or candidates,key=lambda c:(c[0],c[1],-c[2]))
        threshold=chosen[2]
        raw=pipe.predict(test[COLS]); pred,flag,probs,classes=predict_with_gate(pipe,test,threshold)
        report=classification_report(test.human_label,raw,labels=LABELS,output_dict=True,zero_division=0)
        accepted=pred=='same';tp=int(((test.human_label.to_numpy()=='same')&accepted).sum())
        safe_precision=tp/int(accepted.sum()) if accepted.sum() else None
        results[name]={'accuracy':float(accuracy_score(test.human_label,raw)),
                       'macro_f1':float(report['macro avg']['f1-score']),
                       'same_precision_raw':float(report['same']['precision']),
                       'same_precision_auto_accepted':safe_precision,
                       'same_auto_accepted':int(accepted.sum()),'same_auto_correct':tp,
                       'manual_review_count':int(flag.sum()),'threshold':threshold,
                       'validation_threshold_candidates':[{'precision':p,'true_positives':tp,'threshold':t,'accepted':n} for p,tp,t,n in candidates],
                       'classification_report':report,'confusion_matrix':confusion_matrix(test.human_label,raw,labels=LABELS).tolist()}
        output=test[['product_id_a','product_id_b','name_a','name_b','human_label']].copy()
        output['raw_prediction']=raw;output['safe_prediction']=pred;output['needs_review']=flag
        for i,c in enumerate(classes):output['prob_'+c]=probs[:,i]
        output.to_csv(OUT/f'{name}_predictions.csv',index=False,encoding='utf-8-sig')
        joblib.dump(pipe,OUT/f'{name}.joblib');fitted[name]=pipe
        print(name,'accuracy=',round(results[name]['accuracy'],3),'macro-F1=',round(results[name]['macro_f1'],3),
              'auto same precision=',safe_precision,'auto same=',int(accepted.sum()),'review=',int(flag.sum()))
    # Select by validation performance, NOT test results.
    selected=max(results,key=lambda n:(max((c['precision'] for c in results[n]['validation_threshold_candidates'] if c['true_positives']>=3),default=0),results[n]['validation_threshold_candidates'][0]['true_positives']))
    joblib.dump(fitted[selected],OUT/'product_matching_model.joblib')
    summary={'selected_model':selected,'total_pairs':len(df),'train_count':len(train),'validation_count':len(val),'test_count':len(test),
             'split_method':'product_id_disjoint_three_way','note':'Threshold tuned on validation; test used for final reporting. No database writes. Low-confidence same routed to review.',**results}
    (OUT/'comparison_metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Готово:',OUT)
if __name__=='__main__':main()
