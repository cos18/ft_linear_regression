import sys
from ft_linear_regression.util import get_theta, calc_predict

def run():
  sys.tracebacklimit = 0

  mileage = 0
  while True:
    try:
      print('Type mileage which you want to predict price : ', end='')
      mileage = float(input())
    except ValueError:
      print('You need to input numbers!')
      continue
    except OSError as e:
      print(e)

    try:
      print('Estimate price : {}'.format(calc_predict(mileage)))
      print('Using theta : {}'.format(get_theta()))
    except OSError as e:
      print(e)
      
    break
