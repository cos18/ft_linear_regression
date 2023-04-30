from typing import List
import sys
from ft_linear_regression.util import get_theta, get_data, reset_theta, calc_predict, set_theta

def train():
  lr = 0.000001
  while True:
    try:
      print('\nType learning rate using at gradient descent algorithm')
      print('default value is 0.000001\n')
      print('> ', end='')
      lr_input = input()
      if (lr_input == ''):
        break
      lr = float(lr_input)
    except ValueError:
      print('You need to input numbers!')
      continue
    break
  theta = get_theta()
  data = get_data()

  def calc_gradient() -> List[float]:
    result = [0, 0]
    for d in data:
      diff = (calc_predict(theta, d[0]) - d[1]) * lr / len(data)
      result[0] += diff
      result[1] += (diff * d[0])
    return result

  for times in range(1, 10001):
    gradient = calc_gradient()
    theta = [ theta[i] - lr * gradient[i] / len(data) for i in range(2) ]
    if times % 2000 == 0:
      print("{}th try theta : {}".format(times, theta))

  set_theta(theta)
  print("Train complete!!")

  def calc_mse() -> float:
    return sum([(d[1] - calc_predict(theta, d[0])) ** 2 / len(data) for d in data])

  print("MSE of train theta {} is {}".format(theta, calc_mse()))




def print_menu():
  print('\nChoose the menu of train program!\n')
  print('t: Train Model with data!')
  print('r: Reset theta to initial condition')
  print('m: Re-Print the menu of train program')
  print('e: Exit the program\n')

def run():
  sys.tracebacklimit = 0

  print_menu()

  while True:
    print('> ', end='')

    menu = input()

    if menu == 't':
      train()
      continue
    if menu == 'r':
      reset_theta()
      print('Reset theta complete!\n')
      continue
    if menu == 'm':
      print_menu()
      continue
    if menu == 'e':
      break

    print('train: menu not found')
    print('Type "m" to re-print menu\n')