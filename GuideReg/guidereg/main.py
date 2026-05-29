from pprint                    import pprint
from guidereg.cli.parser       import build_parser
from guidereg.configs.loader   import load_config
from guidereg.utils.logger     import setup_logger
from guidereg.utils.validation import validate_config
from guidereg.utils.filesystem import create_directory
from guidereg.pipelines.pipeline import run_pipeline
from guidereg.pipelines.propagation_pipeline import run_propagation_pipeline

def main():

    logger = setup_logger()
    logger.info("Starting GuideReg")

    # ==========================================
    # PARSE CLI
    # ==========================================
    parser = build_parser()
    args   = parser.parse_args()
    logger.info(f"Selected task: {args.task}")

    # ==========================================
    # LOAD CONFIG
    # ==========================================
    logger.info("Loading configuration")
    config = load_config(args.config)

    # ==========================================
    # VALIDATE CONFIG
    # ==========================================
    logger.info("Validating configuration")
    validate_config(config, args.task)

    # ==========================================
    # CREATE OUTPUT DIRECTORY
    # ==========================================
    output_dir = config["output"]["directory"]
    create_directory(output_dir)
    logger.info(f"Output directory ready: {output_dir}")

    # ==========================================
    # PRINT CONFIG
    # ==========================================
    logger.info("Configuration loaded successfully")
    print("\n========== GUIDEREG ==========\n")

    # ==========================================
    # TASK DISPATCH
    # ==========================================
    if args.task == "pipeline":
        run_pipeline(
            config,
            logger
        )

    elif args.task == "propagate":
        run_propagation_pipeline(
            config,
            logger
        )

    else:
        raise ValueError(
            f"Unknown task: {args.task}"
        )

    print("\n==============================\n")


if __name__ == "__main__":
    main()