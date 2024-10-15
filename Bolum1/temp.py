# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Kütüphaneleri import etme
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Veri yükleme

veriler = pd.read_csv('veriler.csv')

#veri ön işleme
print(veriler)

boy = veriler[['boy']]

print(boy)

boykilo = veriler[['boy','kilo']]
print(boykilo)

class insan:
    boy=180
    def kosmak(self,b):
        return b+10;

ali = insan()

print(ali.boy)
print(ali.kosmak(7))