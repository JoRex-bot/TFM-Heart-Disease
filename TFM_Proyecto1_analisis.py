#!/usr/bin/env python3
"""TFM MIA - Clasificacion binaria de enfermedad cardiaca con UCI Cleveland.

Uso (colocar heart.csv en la misma carpeta):
    python TFM_Proyecto1_analisis.py

Dependencias: numpy, matplotlib, scikit-learn.
Todos los ajustes y transformaciones se estiman SOLO sobre datos de entrenamiento.
"""
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
CSV_FILE = ROOT / 'heart.csv'
OUT = ROOT / 'Figuras_TFM'
OUT.mkdir(exist_ok=True)
FEATURES = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach',
            'exang','oldpeak','slope','ca','thal']
NUMERIC = ['age','trestbps','chol','thalach','oldpeak']
CATEGORICAL = ['sex','cp','fbs','restecg','exang','slope','ca','thal']
MISSING = {'','?','NA','NaN','nan','NULL','null'}


def load_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError('El CSV esta vacio')
    needed = FEATURES + ['num']
    if set(needed) != set(rows[0]):
        raise ValueError(f'Columnas inesperadas: {list(rows[0])}')
    missing = {k: sum(str(r[k]).strip() in MISSING for r in rows) for k in needed}
    if any(missing.values()):
        raise ValueError(f'Valores faltantes detectados (requieren imputacion): {missing}')
    data = np.array([[float(r[c]) for c in FEATURES] for r in rows], dtype=float)
    labels = np.array([int(float(r['num'])) for r in rows], dtype=int)
    return rows, data, labels


def create_preprocessor(for_logistic=True):
    ncols = [FEATURES.index(c) for c in NUMERIC]
    ccols = [FEATURES.index(c) for c in CATEGORICAL]
    num_processor = StandardScaler() if for_logistic else 'passthrough'
    return ColumnTransformer([('num',num_processor,ncols),
                              ('cat',OneHotEncoder(handle_unknown='ignore'),ccols)])


def save_fig(name):
    plt.tight_layout()
    plt.savefig(OUT/name, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def descriptives(x, labels):
    y = (labels > 0).astype(int)
    stats = {}
    for feat in NUMERIC:
        v = x[:,FEATURES.index(feat)]
        stats[feat] = {'media':round(float(v.mean()),2),'mediana':round(float(np.median(v)),2),
                       'desv_est':round(float(np.std(v,ddof=1)),2),
                       'min':float(v.min()),'max':float(v.max()),
                       'media_sanos':round(float(v[y==0].mean()),2),
                       'media_enfermos':round(float(v[y==1].mean()),2)}
    return stats


def make_eda(x, labels):
    y = (labels > 0).astype(int)
    counts = np.bincount(y,minlength=2)
    fig,ax=plt.subplots(figsize=(7,4.6))
    bars = ax.bar(['Sin enfermedad','Con enfermedad'], counts)
    ax.set(title='Figura 1. Distribución de la variable objetivo',ylabel='Número de pacientes',ylim=(0,190))
    for b,n in zip(bars,counts):
        ax.annotate(f'{n} ({n/len(y)*100:.1f} %)',(b.get_x()+b.get_width()/2,b.get_height()),
                    xytext=(0,5),textcoords='offset points',ha='center',fontsize=10)
    save_fig('01_distribucion_objetivo.png')

    age=x[:,FEATURES.index('age')]
    fig,ax=plt.subplots(figsize=(8,4.7))
    ax.hist(age[y==0],bins=np.arange(25,85,5),alpha=.65,label='Sin enfermedad')
    ax.hist(age[y==1],bins=np.arange(25,85,5),alpha=.65,label='Con enfermedad')
    ax.set(title='Figura 2. Distribución de edad según enfermedad',xlabel='Edad (años)',ylabel='Número de pacientes')
    ax.legend()
    save_fig('02_edad_por_grupo.png')

    cp=x[:,FEATURES.index('cp')].astype(int)
    cp_classes=np.array([1,2,3,4]);pos=np.array([sum(y[cp==val]) for val in cp_classes]);tot=np.array([sum(cp==val) for val in cp_classes])
    fig,ax=plt.subplots(figsize=(8,4.8))
    bars=ax.bar(['1: típica','2: atípica','3: no anginosa','4: asintomático'],100*pos/tot)
    ax.set(title='Figura 3. Proporción con enfermedad según dolor torácico',xlabel='Tipo de dolor torácico (cp)',ylabel='Enfermedad presente (%)',ylim=(0,100))
    for b,n,t in zip(bars,pos,tot):
        ax.annotate(f'{n}/{t}',(b.get_x()+b.get_width()/2,b.get_height()),xytext=(0,5),textcoords='offset points',ha='center')
    save_fig('03_enfermedad_por_dolor_toracico.png')

    ca=x[:,FEATURES.index('ca')].astype(int)
    ca_classes=np.array([0,1,2,3]);pos_ca=np.array([sum(y[ca==val]) for val in ca_classes]);tot_ca=np.array([sum(ca==val) for val in ca_classes])
    fig,ax=plt.subplots(figsize=(7.8,4.7))
    bars=ax.bar([f'{n} vasos' for n in ca_classes],100*pos_ca/tot_ca)
    ax.set(title='Figura 4. Proporción con enfermedad según ca',xlabel='Número de vasos mayores coloreados por fluoroscopia',ylabel='Enfermedad presente (%)',ylim=(0,105))
    for b,n,t in zip(bars,pos_ca,tot_ca):
        ax.annotate(f'{n}/{t}',(b.get_x()+b.get_width()/2,b.get_height()),xytext=(0,5),textcoords='offset points',ha='center')
    save_fig('04_enfermedad_por_ca.png')

    hr=x[:,FEATURES.index('thalach')]
    fig,ax=plt.subplots(figsize=(7.4,4.8))
    ax.boxplot([hr[y==0],hr[y==1]],tick_labels=['Sin enfermedad','Con enfermedad'],showmeans=True)
    ax.set(title='Figura 5. Frecuencia cardíaca máxima alcanzada',ylabel='Latidos por minuto (thalach)')
    save_fig('05_frecuencia_maxima.png')
    return {'objetivo':{'sin_enfermedad':int(counts[0]),'con_enfermedad':int(counts[1]),
                        'prevalencia_muestra_pct':round(100*float(y.mean()),2)},
            'dolor_toracico':{str(v):{'n':int(t),'n_enfermos':int(p),'pct_enfermos':round(100*float(p/t),2)} for v,t,p in zip(cp_classes,tot,pos)},
            'vasos_ca':{str(v):{'n':int(t),'n_enfermos':int(p),'pct_enfermos':round(100*float(p/t),2)} for v,t,p in zip(ca_classes,tot_ca,pos_ca)}}


def evaluate_model(name, model, train_x, test_x, train_y, test_y):
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    cvscores=cross_validate(model,train_x,train_y,cv=cv,scoring={'exactitud':'accuracy','auc':'roc_auc','sensibilidad':'recall'},n_jobs=1)
    model.fit(train_x,train_y)
    pred=model.predict(test_x)
    proba=model.predict_proba(test_x)[:,1]
    cm=confusion_matrix(test_y,pred,labels=[0,1]);tn,fp,fn,tp=cm.ravel()
    return ({'exactitud_pct':round(100*accuracy_score(test_y,pred),2),
             'exactitud_balanceada_pct':round(100*balanced_accuracy_score(test_y,pred),2),
             'precision_pct':round(100*precision_score(test_y,pred,zero_division=0),2),
             'sensibilidad_pct':round(100*recall_score(test_y,pred),2),
             'especificidad_pct':round(100*tn/(tn+fp),2),
             'f1_pct':round(100*f1_score(test_y,pred),2),
             'auc_roc':round(float(roc_auc_score(test_y,proba)),4),
             'cv_exactitud_media_pct':round(100*float(cvscores['test_exactitud'].mean()),2),
             'cv_exactitud_sd_pct':round(100*float(cvscores['test_exactitud'].std()),2),
             'cv_auc_media':round(float(cvscores['test_auc'].mean()),4),
             'cv_sensibilidad_media_pct':round(100*float(cvscores['test_sensibilidad'].mean()),2),
             'matriz_confusion':cm.tolist(), 'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp)},
            proba)


def make_model_figures(test_y, results_proba, matrix):
    fig,ax=plt.subplots(figsize=(6.7,5))
    for name,proba in results_proba.items():
        fpr,tpr,_=roc_curve(test_y,proba)
        ax.plot(fpr,tpr,label=f'{name} (AUC {roc_auc_score(test_y,proba):.3f})')
    ax.plot([0,1],[0,1],'--',color='gray',label='Azar')
    ax.set(xlabel='Tasa de falsos positivos',ylabel='Sensibilidad',title='Figura 6. Curvas ROC en conjunto de prueba',xlim=(-.01,1.01),ylim=(-.01,1.01))
    ax.legend(loc='lower right')
    save_fig('06_curvas_ROC.png')
    fig,ax=plt.subplots(figsize=(6.4,4.8))
    disp=ConfusionMatrixDisplay(confusion_matrix=np.array(matrix),display_labels=['Sin enfermedad','Con enfermedad'])
    disp.plot(ax=ax,cmap='Blues',colorbar=False,values_format='d')
    ax.set(title='Figura 7. Matriz de confusión: regresión logística',xlabel='Predicción',ylabel='Resultado real')
    save_fig('07_matriz_confusion_logistica.png')


def generate_report(results):
    d=results['descriptivos'];e=results['EDA'];ms=results['modelos'];
    lines=[
        '# Borrador de resultados del TFM: análisis exploratorio y modelos',
        '', '## 1. Fuente de datos',
        'Archivo facilitado: `heart.csv`, compatible con la estructura de 14 columnas (13 predictores + `num`) del conjunto UCI Heart Disease (Cleveland).',
        'Fuente original: https://archive.ics.uci.edu/dataset/45/heart+disease (consultada el 7 de noviembre de 2026).',
        'No se conoce el proceso previo por el que este CSV llegó a no contener valores faltantes; no debe asumirse que coincide sin cambios con el original de UCI.',
        '', '## 2. Limpieza y definición del resultado',
        f"Se analizaron **{results['registros']} registros**, con {len(FEATURES)} variables predictoras. Se encontraron **{results['valores_faltantes']} celdas faltantes**, **{results['duplicados']} filas duplicadas** y ninguna columna adicional.",
        'Se transformó `num` en una variable binaria: `0` si num=0 (sin enfermedad) y `1` si num toma valores de 1 a 4 (con enfermedad). Esta dicotomización sigue la práctica descrita por UCI.',
        f"Las clases resultantes fueron **{e['objetivo']['sin_enfermedad']} sin enfermedad** y **{e['objetivo']['con_enfermedad']} con enfermedad** ({e['objetivo']['prevalencia_muestra_pct']} % del total).",
        '', '## 3. Resultados exploratorios',
        '| Variable numérica | Media total | Media sin enfermedad | Media con enfermedad | Mínimo–máximo |',
        '|---|---:|---:|---:|---:|',
    ]
    label={'age':'Edad (años)','trestbps':'Presión sistólica en reposo (mmHg)','chol':'Colesterol (mg/dl)','thalach':'FC máxima (lpm)','oldpeak':'Depresión ST (oldpeak)'}
    for k in NUMERIC:
        dd=d[k]
        lines.append(f"| {label[k]} | {dd['media']} | {dd['media_sanos']} | {dd['media_enfermos']} | {dd['min']:g}–{dd['max']:g} |")
    lines += ['', 'Entre las relaciones descriptivas destacan las diferencias de distribución según tipo de dolor torácico (`cp`) y número de vasos visualizados por fluoroscopia (`ca`), mostradas en las figuras 3 y 4. Estas asociaciones son descriptivas y no demuestran causalidad.',
              'La frecuencia cardíaca máxima alcanzada (`thalach`) también presenta distribuciones distintas entre los grupos (figura 5).',
              '', '## 4. Entrenamiento y evaluación',
              f"Se dividió el conjunto mediante muestreo estratificado (semilla 42): **{results['train_n']} registros para entrenamiento** y **{results['test_n']} para prueba**. En los datos de entrenamiento se aplicó validación cruzada estratificada de 5 particiones.",
              'Para regresión logística se estandarizaron las variables numéricas y se codificaron las categóricas mediante one-hot encoding. Random Forest empleó variables numéricas sin estandarizar y la misma codificación categórica. El preprocesamiento se ajustó dentro del pipeline de entrenamiento/validación para evitar fuga de información.',
              '', '| Métrica | Regresión logística | Random Forest |', '|---|---:|---:|']
    metrics=[('Exactitud (%)','exactitud_pct'),('Sensibilidad (%)','sensibilidad_pct'),('Especificidad (%)','especificidad_pct'),('F1 (%)','f1_pct'),('AUC-ROC','auc_roc'),('Exactitud media CV (%)','cv_exactitud_media_pct'),('AUC media CV','cv_auc_media')]
    for desc,key in metrics:
        lines.append(f"| {desc} | {ms['Regresión logística'][key]} | {ms['Random Forest'][key]} |")
    a=ms['Regresión logística']
    lines += ['',f"En el conjunto de prueba, la regresión logística generó **{a['TP']} verdaderos positivos**, **{a['TN']} verdaderos negativos**, **{a['FP']} falsos positivos** y **{a['FN']} falsos negativos** (figura 7).",
              '', '## 5. Limitaciones de interpretación',
              '- Conjunto pequeño (303 casos) y procedencia histórica; no es representativo necesariamente de la práctica clínica actual o la población ecuatoriana.',
              '- Los grupos proceden de una muestra seleccionada; la proporción de enfermedad no debe interpretarse como prevalencia poblacional.',
              '- La variable `ca` requiere información de fluoroscopia; el modelo no debe presentarse como prueba de detección sin considerar la disponibilidad y naturaleza diagnóstica de sus predictores.',
              '- El rendimiento procede de una única partición de prueba y de validación cruzada interna; falta validación externa, calibración y evaluación clínica.',
              '- El CSV entregado no contiene ausencias, aunque UCI describe valores faltantes en la fuente original. La procedencia de su procesamiento debe documentarse.',
              '', '## 6. Figuras disponibles',
              *[f'- `{p.name}`' for p in sorted(OUT.glob('*.png'))],
              '', '## 7. Reproducibilidad',
              'Ejecutar `python TFM_Proyecto1_analisis.py` en la carpeta que contenga `heart.csv` (dependencias: numpy, matplotlib y scikit-learn). Las tablas y figuras se generan automáticamente.',
              'Los resultados no deben interpretarse como evidencia de utilidad clínica ni utilizarse para decisiones asistenciales individuales.',
              '']
    (ROOT/'TFM_Borrador_EDA_y_Modelos.md').write_text('\n'.join(lines),encoding='utf8')


def main():
    rows,x,raw_y=load_csv(CSV_FILE)
    y=(raw_y>0).astype(int)
    duplicate_count=len(rows)-len({tuple(r[k] for k in FEATURES+['num']) for r in rows})
    eda=make_eda(x,raw_y)
    train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=.20,random_state=42,stratify=y)
    models={
        'Regresión logística': Pipeline([('preprocesamiento',create_preprocessor(True)),('clasificador',LogisticRegression(max_iter=3000,random_state=42))]),
        'Random Forest': Pipeline([('preprocesamiento',create_preprocessor(False)),('clasificador',RandomForestClassifier(n_estimators=300,min_samples_leaf=2,random_state=42,n_jobs=1))]),
    }
    metrics,probas={},{}
    for name,model in models.items():
        metrics[name],probas[name]=evaluate_model(name,model,train_x,test_x,train_y,test_y)
    make_model_figures(test_y,probas,metrics['Regresión logística']['matriz_confusion'])
    stats={'registros':len(rows),'predictores':len(FEATURES),'valores_faltantes':0,'duplicados':duplicate_count,
           'train_n':len(train_y),'test_n':len(test_y),
           'objetivo_original':dict(Counter(map(str,raw_y.tolist()))),
           'descriptivos':descriptives(x,raw_y),'EDA':eda,'modelos':metrics,
           'configuracion':{'semilla':42,'prueba':'20 % estratificado','validacion_cruzada':'5 folds, solo entrenamiento',
                            'umbral_clasificacion':0.5}}
    (ROOT/'TFM_Resultados_EDA.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf8')
    generate_report(stats)
    print(json.dumps(stats,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
