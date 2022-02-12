## Problem Statement

Generate a dataset of 400 2-dimensional data vectors which consists of four groups of 100 data vectors. The four groups are modeled by Gaussian distributions with means ```m1 = [0,0]```, ```m2 = [4,0]```, ```m3 = [0,4]```, ```m4 = [5,4]``` respectively, covariance matrices 
```S1 = I```, ```S2 = [[1, 0.2], [0.2, 1.5]]```, ```S3 = [[1, 0.4], [0.4, 1.1]]```, ```S4 = [[0.3, 0.2], [0.2, 0.5]]``` respectively. Plot the data vectors. Measures the Euclidean distance between any two data points and determine maximum (```dmax```) and minimum (```dmin```) Euclidean distances.

 
- Create virtual environment install dependencies by  
```
virtualenv venv && pip install -r requirements.txt
```