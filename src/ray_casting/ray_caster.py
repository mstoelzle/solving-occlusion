import faulthandler; faulthandler.enable()
import numpy as np
import os


class RayCaster:
    def __init__(self):
        import raisimpy as raisim
        self.raisim = raisim
        raisim.World.setLicenseFile(os.getcwd() + "/../raisim-license/activation_Maximilian_Stoelzle_macos.raisim")

        self.dem = None
        self.res_grid = None
        self.max_ray_cast_dist = 0.0

    def load_dem(self, dem: np.array, res_grid: np.array):
        self.dem = dem
        self.res_grid = res_grid
        self.max_ray_cast_dist = (np.max(res_grid) * np.max(dem.shape)).item()

        center_x, center_y = 0., 0.
        height_vec = dem.copy().reshape((-1, ), order="F")

        self.world = self.raisim.World()
        self.world.addHeightMap(dem.shape[0], dem.shape[1], dem.shape[0]*res_grid[0], dem.shape[1]*res_grid[1],
                                center_x, center_y, height_vec)

        # self.server = self.raisim.RaisimServer(self.world)
        # self.server.integrateWorldThreadSafe()
        # self.server.launchServer(8080)

    def run(self, rel_position: np.array):
        print("max_ray_cast_dist", self.max_ray_cast_dist)

        ray_direction = np.array([0, 0, 0], dtype=np.float64)

        print(self.world.rayTest(rel_position, ray_direction, self.max_ray_cast_dist))
