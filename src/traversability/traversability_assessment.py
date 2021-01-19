import numpy as np
import torch


from src.enums import *

class TraversabilityAssessment:
    def __init__(self, **config: dict):
        self.config = config

        from traversability_pybind import Traversability
        self.traversability = Traversability()

        # self.traversability.configure_traversability()

    def __call__(self, output: dict[ChannelEnum, torch.Tensor],
                 data: dict[ChannelEnum, torch.Tensor]) -> dict[ChannelEnum, torch.Tensor]:
        rec_dems = output[ChannelEnum.REC_DEM]
        comp_dems = output[ChannelEnum.COMP_DEM]
        res_grid = data[ChannelEnum.RES_GRID]

        rec_trav_risk_maps = []
        comp_trav_risk_maps = []
        for idx in range(rec_dems.size(0)):
            rec_dem = rec_dems[idx, ...].detach().cpu().numpy()
            comp_dem = comp_dems[idx, ...].detach().cpu().numpy()

            # Defaults from HDPR config:
            # https://github.com/esa-prl/bundles-rover/blob/master/config/orogen/traversability::Task.yml
            # Size of the kernel used in the laplacian to detect rocks edges
            laplacian_kernel_size = int(self.config.get("laplacian_kernel_size", 9))
            # Threshold of the laplacian to detect rocks edges. It is not zet tied to a physical property.
            laplacian_threshold = self.config.get("laplacian_threshold", 100)
            # First obstacle dilation iterations
            obstacle_iterations = int(self.config.get("obstacle_iterations", 2))
            # Kernel size to dilate obstacle first time [cells]
            obstacle_kernel_size = int(self.config.get("obstacle_kernel_size", 3))
            # Second obstacle dilation iterations (area surrounding an obstacle)
            obstacle_vicinity_iterations = int(self.config.get("obstacle_vicinity_iterations", 3))
            # Kernel size to dilate obstacle second time [cells]
            obstacle_vicinity_kernel_size = int(self.config.get("obstacle_vicinity_kernel_size", 3))
            # in meters, needed for obstacle dilation
            robot_size = self.config.get("robot_size", 0.6)
            # Obstacles the rover can tolerate when there is no sinkage [m]
            rover_obstacle_clearance = self.config.get("rover_obstacle_clearance", 0.2)
            # Slope the rover can travel in nominal case [rad]
            rover_slope_gradeability = self.config.get("rover_slope_gradeability", 15/180*np.pi)
            # How many normal cell a slope map cell will contain
            slope_map_scale = int(self.config.get("slope_map_scale", 3))
            # Final traversability map dilation iterations
            dilation_iterations = int(self.config.get("dilation_iterations", 2))

            # Resolution of each cell in meters
            assert res_grid[idx, 0] == res_grid[idx, 1]
            map_resolution = res_grid[idx, 0]

            self.traversability.configure_traversability(map_resolution, slope_map_scale,
                                                         rover_slope_gradeability, rover_obstacle_clearance,
                                                         laplacian_kernel_size, laplacian_threshold,
                                                         obstacle_kernel_size, obstacle_iterations,
                                                         obstacle_vicinity_kernel_size, obstacle_vicinity_iterations,
                                                         robot_size, dilation_iterations)

            print("newshape", rec_dem.reshape((-1), order='C').shape)

            # the perception-traversability module expects a std::vector in row-major format
            # https://pybind11.readthedocs.io/en/stable/advanced/cast/overview.html
            self.traversability.set_elevation_map(rec_dem.reshape((-1), order='C').tolist(),
                                                  rec_dem.shape[0], rec_dem.shape[1])
            rec_trav_risk_map = np.array(self.traversability.compute_traversability())

            print("rec_trav_risk_map", rec_trav_risk_map)

            self.traversability.set_elevation_map(comp_dem.reshape((-1), order='C').tolist(),
                                                  comp_dem.shape[0], comp_dem.shape[1])
            comp_trav_risk_map = np.array(self.traversability.compute_traversability())

            rec_trav_risk_maps.append(torch.tensor(rec_trav_risk_map))
            comp_trav_risk_maps.append(torch.tensor(comp_trav_risk_map))

        output[ChannelEnum.REC_TRAV_RISK_MAP] = torch.stack(rec_trav_risk_maps)
        output[ChannelEnum.COMP_TRAV_RISK_MAP] = torch.stack(comp_trav_risk_maps)

        return output
