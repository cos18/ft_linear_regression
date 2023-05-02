import sys
import matplotlib.pyplot as plt
import numpy as np

from ft_linear_regression.plot import plot_line
from ft_linear_regression.util.file import get_model_info, get_data, reset_model_info, set_model_info
from ft_linear_regression.util.calc import calc_gradient, calc_mse, calc_array_norm


def train():
    model_info = get_model_info()
    x_km, y_price = get_data()
    model_info[2] = x_km.mean()
    model_info[3] = x_km.std()
    mse = calc_mse(model_info, x_km, y_price)
    lr = 0.01

    plt.ion()
    fig, ax = plt.subplots()
    line = None
    ax.plot(x_km, y_price, 'o', color='tab:brown')

    ax.set_xlabel('km')
    ax.set_ylabel('price')

    fig.canvas.draw()
    fig.canvas.flush_events()

    while True:
        try:
            print('\nType learning rate using at gradient descent algorithm')
            print('default value is 0.01')
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

    for times in range(1, 1001):
        gradient = calc_gradient(model_info, x_km, y_price)
        update_model_info = np.concatenate([model_info[0:2] - lr * gradient, model_info[2:4]], axis=None)
        update_mse = calc_mse(update_model_info, x_km, y_price)

        if update_mse > mse:
            print("Learning Rate is TOO HIGH!!!")
            print("Canceling Training...")
            plt.close(fig)
            return
        elif mse - update_mse < 0.000001:
            print('Train is enough to run more...')
            print('Finishing Training...\n')
            break

        model_info = update_model_info
        mse = update_mse
        if times % 100 == 0:
            line = plot_line(ax, x_km, model_info, line)
            fig.canvas.draw()
            fig.canvas.flush_events()

            print("{}th try theta : {} / MSE : {}".format(times, model_info[0:2], mse))
            input("Press enter to continue...")

    set_model_info(model_info)
    plot_line(ax, x_km, model_info, line)
    fig.canvas.draw()
    fig.canvas.flush_events()
    print("Train complete!!")

    print("MSE of train theta {} is {}".format(model_info[0:2], mse))
    input("Press enter to continue...")
    plt.close('all')


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
            reset_model_info()
            print('Reset theta complete!\n')
            continue
        if menu == 'm':
            print_menu()
            continue
        if menu == 'e':
            break

        print('train: menu not found')
        print('Type "m" to re-print menu\n')
