import argparse


def build_parser():

    parser = argparse.ArgumentParser(
        prog="GuideReg",
        description="Guided Medical image registration framework"
    )

    subparsers = parser.add_subparsers(
        dest="task",
        required=True
    )

    # ==================================================
    # INITIAL REGISTRATION
    # ==================================================
    initial_parser = subparsers.add_parser(
        "initial",
        help="Run initial registration"
    )

    initial_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )

    # ==================================================
    # RIGID REGISTRATION
    # ==================================================
    rigid_parser = subparsers.add_parser(
        "rigid",
        help="Run rigid registration"
    )

    rigid_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )

    # ==================================================
    # FULL PIPELINE
    # ==================================================
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run full registration pipeline"
    )

    pipeline_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )

    # ==================================================
    #  PROPAGATION FOR SEGMENTATIONS
    # ==================================================
    propagation_parser = subparsers.add_parser(
        "propagate",
        help="Propagate segmentations using existing transforms"
    )

    propagation_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )

    return parser