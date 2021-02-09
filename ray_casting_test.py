import numpy as np


from src.ray_casting.ray_caster import RayCaster


def main():
    ray_casting = RayCaster()

    sample_dem = np.zeros(shape=(64, 64))
    ray_casting.load_dem(sample_dem, np.array([0.05, 0.05]))
    ray_casting.run([0, 0, 1])


if __name__ == '__main__':
    main()
