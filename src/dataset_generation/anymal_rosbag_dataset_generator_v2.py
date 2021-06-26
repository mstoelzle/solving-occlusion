from .rosbag_dataset_generator import RosbagDatasetGenerator


class AnymalRosbagDatasetGenerator(RosbagDatasetGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.ros_topics = ['/elevation_mapping/elevation_map_recordable']
