# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 09:11:01 2026

@author: Ale
"""

import platform
import netket as nk
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import scipy
import matplotlib

print("Python:", platform.python_version())
print("NetKet:", nk.__version__)
print("JAX:", jax.__version__)
print("Flax:", nn.__name__)
print("NumPy:", np.__version__)
print("SciPy:", scipy.__version__)
print("Matplotlib:", matplotlib.__version__)

# Verifica i dispositivi disponibili
print("Dispositivi JAX:", jax.devices())

# Esegue un semplice calcolo JAX
x = jnp.array([1.0, 2.0, 3.0])
y = jnp.sum(x**2)

print("Risultato del calcolo JAX:", y)

# Forza l'esecuzione del calcolo
print("Valore numerico:", float(y))

print("\nTest completato!")