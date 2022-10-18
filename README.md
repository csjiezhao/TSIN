### Temporal Spatial Intertwined Network(TSIN)

> This repository is the implementation of **TSIN** in Pytorch. Please note that we only provide the basic framework of STIN for now, and the core code will be made public after the paper is published.

#### Data

- NYC-Taxi & FHV: https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- CHI-Taxi: https://data.cityofchicago.org/Transportation/Transportation-Network-Providers-Trips-2018/rpfj-eb3a
- CHI-TNP: https://data.cityofchicago.org/Transportation/Transportation-Network-Providers-Trips/m6dm-c72p

Pre-processing of trip data will be released later...

#### Requirements

- Pytorch >= 1.8
- Numpy

#### Usage
```python
# train the TSIN model
run train.py
# test the TSIN model
run test.py
```
