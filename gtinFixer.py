#!/usr/bin/python3
import numpy as np

def valida_gtin(gtin):
    '''
    :param gtin:
    :return:
    '''
    if len(gtin) == 14:
        gtin = gtin_14_para_13(gtin)
    vec_mult = np.array([1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3])
    if len(gtin) == 13:
        vec_gtin = np.array(list(gtin[0:12]))
        produto_escalar = np.sum(vec_mult * vec_gtin.astype(int))
        prox_decimal = int(produto_escalar / 10) * 10 + 10
        resultado = prox_decimal - produto_escalar
        resultado = resultado % 10
        if str(resultado) == gtin[12]:
            return str(gtin).zfill(13)
    return np.nan

def gtin_14_para_13(gtin14):
    '''
    Converte um GTIN de 14 dígitos no equivalente de 13 dígitos
    :param gtin14: gtin de 14 dígitos
    :return: gtin de 13 dígitos
    '''
    vec_mult = np.array([1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3])
    gtin_13 = np.array(
        [gtin14[1], gtin14[2], gtin14[3], gtin14[4], gtin14[5], gtin14[6], gtin14[7], gtin14[8], gtin14[9], gtin14[10],
         gtin14[11], gtin14[12]])
    produto_escalar = np.sum(vec_mult * gtin_13.astype(int))
    prox_decimal = int(produto_escalar / 10) * 10 + 10
    resultado = prox_decimal - produto_escalar
    resultado = resultado % 10
    gtin_13 = np.append(gtin_13, resultado)
    return ''.join(gtin_13)
