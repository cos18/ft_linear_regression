import sys
import numpy as np
import matplotlib.pyplot as plt

from ft_linear_regression.plot import plot_line
from ft_linear_regression.util.file import get_theta, get_data, reset_theta, set_theta
from ft_linear_regression.util.calc import calc_gradient, calc_mse


def train():
    theta = get_theta()
    x_km, y_price = get_data()
    print(x_km, y_price)
    lr = 0.000000001
    mse = calc_mse(theta, x_km, y_price)

    plt.ion()
    fig, ax = plt.subplots()
    ax.plot(x_km, y_price, 'o', color='tab:brown')
    line = plot_line(ax, x_km, theta)

    ax.set_xlabel('km')
    ax.set_ylabel('price')

    fig.canvas.draw()
    fig.canvas.flush_events()

    while True:
        try:
            print('\nType learning rate using at gradient descent algorithm')
            print('default value is 0.000001')
            print('If lr is too high, program will cancel training\n')
            print('> ', end='')
            lr_input = input()
            if lr_input == '':
                break
            lr = float(lr_input)
        except ValueError:
            print('You need to input numbers!')
            continue
        break

    for times in range(1, 10001):
        gradient = calc_gradient(theta, x_km, y_price)
        print(gradient)
        update_theta = theta - lr * gradient
        update_mse = calc_mse(update_theta, x_km, y_price)

        if update_mse > mse:
            print("Learning Rate is TOO HIGH!!!")
            print("Canceling Training...")
            plt.close(fig)
            return

        theta = update_theta
        mse = update_mse
        if times % 1000 == 0:
            line = plot_line(ax, x_km, theta, line)
            fig.canvas.draw()
            fig.canvas.flush_events()

            print("{}th try theta : {}".format(times, theta))
            input()

    set_theta(theta)
    plot_line(ax, x_km, theta, line)
    fig.canvas.draw()
    fig.canvas.flush_events()
    print("Train complete!!")

    print("MSE of train theta {} is {}".format(theta, mse))
    input()
    plt.close(fig)


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
