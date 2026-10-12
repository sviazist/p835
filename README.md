# p835

Python-реализация **ITU-R P.835-7** — «Reference standard
atmospheres».

## Установка

```bash
pip install -e .
```

## Использование

```python
from p835 import global_temperature, global_pressure

T = global_temperature(10.0)   # K
P = global_pressure(10.0)      # гПа
```

## Тесты

```bash
pytest tests/ -v
```

## Attribution

ATTRIBUTION AND SOURCES
========================

This software (PyP835) is a Python implementation of the mathematical
model described in:

    ITU-R P.835-7 — "Reference standard atmospheres"
    International Telecommunication Union, Radiocommunication Sector (ITU-R)
    Geneva, Switzerland

The ITU-R Recommendations are publicly available at:
    https://www.itu.int/rec/R-REC-P.835-7/en

The original reference implementation in C++ was published by the
U.S. National Telecommunications and Information Administration (NTIA)
and is available at:
    https://github.com/NTIA/p835-7

The Python port was created by:
    Evgeny Gennadievich Prutkov <sviazist.pr@gmail.com>, 2026

This software is provided under the MIT License (see LICENSE).

The author of this port is not affiliated with ITU or NTIA. All
formulas and algorithms are taken from the publicly available
ITU-R Recommendations.
