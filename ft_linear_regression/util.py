from typing import List

def get_theta() -> List[int]:
  try:
    with open('ft_linear_regression/data/theta', 'r') as theta_file:
      theta_org = theta_file.readline().split()
      try:
        return list(map(float, theta_org))
      except ValueError:
        raise ValueError('Error while parsing theta!')
  except OSError:
    raise OSError('Error when opening theta file!')

def set_theta(theta: List[int]):
  try:
    with open('ft_linear_regression/data/theta', 'w') as theta_file:
      theta_file.write(' '.join(theta))
  except OSError:
    raise OSError('Error when writing theta file!')

def reset_theta():
  set_theta([0, 0])

def calc_predict(mileage: float) -> float:
  theta = get_theta()
  return theta[0] + theta[1] * mileage
