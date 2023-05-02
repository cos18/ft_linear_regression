from typing import List

def get_data() -> List[List[float]]:
  try:
    with open('ft_linear_regression/data/data.csv', 'r') as data_file:
      data_file.readline()

      result = []
      data_org = data_file.readline()

      try:
        while data_org != '':
          result.append(list(map(float, data_org.split(','))))
          data_org = data_file.readline()
      except ValueError:
        raise ValueError('Error while parsing data!')

      return result
  except OSError:
    raise OSError('Error when opening data file!')

def get_theta() -> List[float]:
  try:
    with open('ft_linear_regression/data/theta', 'r') as theta_file:
      theta_org = theta_file.readline().split()
      try:
        return list(map(float, theta_org))
      except ValueError:
        raise ValueError('Error while parsing theta!')
  except OSError:
    raise OSError('Error when opening theta file!')

def set_theta(theta: List[float]):
  try:
    with open('ft_linear_regression/data/theta', 'w') as theta_file:
      theta_file.write(' '.join(list(map(str, theta))))
  except OSError:
    raise OSError('Error when writing theta file!')

def reset_theta():
  set_theta([0, 0])

def calc_predict(theta: List[float], mileage: float) -> float:
  return theta[0] + theta[1] * mileage

def calc_mse(theta: List[float], data: List[List[float]]) -> float:
  return sum([(d[1] - calc_predict(theta, d[0])) ** 2 / len(data) for d in data])
