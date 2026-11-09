#!/usr/bin/env python3
"""Verificación reproducible de las salidas del TFM (sin dependencias adicionales).

Ejecutar después de `python TFM_Proyecto1_analisis.py`.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'TFM_Resultados_EDA.json'
FIGURES = ROOT / 'Figuras_TFM'


def comprobar(condicion, explicacion):
    if not condicion:
        raise AssertionError(explicacion)
    print(f'OK - {explicacion}')


def main():
    comprobar(RESULTS.is_file(), 'Archivo de resultados JSON generado')
    resultados = json.loads(RESULTS.read_text(encoding='utf-8'))
    for clave, esperado in {'registros': 303, 'predictores': 13, 'train_n': 242,
                            'test_n': 61, 'valores_faltantes': 0, 'duplicados': 0}.items():
        comprobar(resultados.get(clave) == esperado, f'{clave} = {esperado}')
    modelos = resultados['modelos']
    logistica = modelos['Regresión logística']
    bosque = modelos['Random Forest']
    comprobar(logistica['matriz_confusion'] == [[28, 5], [2, 26]],
              'Matriz de confusión de regresión logística = [[28, 5], [2, 26]]')
    for nombre, valor_observado, valor_esperado in (
        ('Exactitud regresión logística', logistica['exactitud_pct'], 88.52),
        ('Sensibilidad regresión logística', logistica['sensibilidad_pct'], 92.86),
        ('Especificidad regresión logística', logistica['especificidad_pct'], 84.85),
        ('AUC regresión logística', logistica['auc_roc'], 0.9665),
        ('Exactitud Random Forest', bosque['exactitud_pct'], 86.89),
        ('AUC Random Forest', bosque['auc_roc'], 0.9502),
    ):
        comprobar(abs(valor_observado - valor_esperado) <= 0.0001,
                  f'{nombre} = {valor_esperado}')
    graficas = list(FIGURES.glob('*.png')) if FIGURES.is_dir() else []
    comprobar(len(graficas) == 7, 'Se generaron las siete figuras esperadas')
    comprobar((ROOT / 'TFM_Borrador_EDA_y_Modelos.md').is_file(),
              'Informe de análisis generado')
    print('\nTODAS LAS VERIFICACIONES DEL TFM FUERON CORRECTAS.')


if __name__ == '__main__':
    main()
