from ft_linear_regression.util import get_theta

def calc_predict(mileage: float) -> float:
  theta = get_theta()
  return theta[0] + theta[1] * mileage

def run():
  while True:
    try:
      print('Type mileage which you want to predict price : ', end='')
      mileage = float(input())
      print('Estimate price : {}'.format(calc_predict(mileage)))
      print('Using theta : {}'.format(get_theta()))
      break
    except:
      print('You need to input numbers!')
