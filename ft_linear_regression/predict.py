import sys
from ft_linear_regression.util.file import get_model_info
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
        model_info = get_model_info()
        print('Estimate price : {}'.format(calc_predict(model_info, mileage)))
        print('Using theta : {}'.format(model_info[0:2]))
    except Exception as e:
        print(e)
