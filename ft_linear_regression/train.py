import sys
from ft_linear_regression.util import reset_theta

def train():
  pass

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