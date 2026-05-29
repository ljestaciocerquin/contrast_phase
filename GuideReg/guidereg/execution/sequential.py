def run_sequential(
    cases,
    pipeline_function,
    config,
    logger
):
    logger.info(
        "Running sequential execution"
    )

    for case in cases:
        pipeline_function(
            case,
            config,
            logger
        )