import sys
from ft_linear_regression.util.file import get_theta
from ft_linear_regression.util.calc import calc_predict


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
        break

    try:
        theta = get_theta()
        print('Estimate price : {}'.format(calc_predict(theta, mileage)))
        print('Using theta : {}'.format(theta))
    except Exception as e:
        print(e)
