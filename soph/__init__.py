import os

configs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")
point_clouds_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "ground_truth_pointclouds"
)
logs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")

DEFAULT_FOOTPRINT_RADIUS = 0.26
