import sys
from ft_linear_regression.util.file import get_theta, get_data, reset_theta, set_theta
from ft_linear_regression.util.calc import calc_gradient, calc_mse

def train():
  lr = 0.000001
  while True:
    try:
      print('\nType learning rate using at gradient descent algorithm')
      print('default value is 0.000001')
      print('If lr is too high, program will cancel training\n')
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
  mse = calc_mse(theta, data)

  for times in range(1, 10001):
    gradient = calc_gradient(theta, data, lr)
    update_theta = [ theta[i] - lr * gradient[i] / len(data) for i in range(2) ]
    update_mse = calc_mse(update_theta, data)

    if (update_mse > mse):
      print("Learning Rate is TOO HIGH!!!")
      print("Calceling Training...")
      return

    theta = update_theta
    mse = update_mse
    if times % 2000 == 0:
      print("{}th try theta : {}".format(times, theta))

  set_theta(theta)
  print("Train complete!!")
  
  print("MSE of train theta {} is {}".format(theta, mse))




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