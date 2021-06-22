from pathlib import Path

from rosbags.typesys import get_types_from_idl, get_types_from_msg, register_types


def register_msg_types():
    msg_type_paths = {
        "grid_map_msgs/msg/GridMap": "ros_msgs/grid_map_msgs/msg/GridMap.msg",
        "msg/GridMapInfo": "ros_msgs/grid_map_msgs/msg/GridMapInfo.msg"
    }

    # plain dictionary to hold message definitions
    add_types = {}

    for msg_type, msg_path in msg_type_paths.items():
        msg_text = (Path(__file__).parent / Path(msg_path)).read_text()

        # add definition from one msg file
        add_types.update(get_types_from_msg(msg_text, msg_type))

    # make types available to rosbags serializers/deserializers
    register_types(add_types)
