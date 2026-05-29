from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed
)


def run_multiprocessing(
    cases,
    pipeline_function,
    config,
    logger,
    num_workers=4
):

    logger.info(
        f"Running multiprocessing "
        f"with {num_workers} workers"
    )

    with ProcessPoolExecutor(
        max_workers=num_workers
    ) as executor:

        futures = [

            executor.submit(
                pipeline_function,
                case,
                config,
                None
            )

            for case in cases
        ]

        for future in as_completed(
            futures
        ):

            try:

                future.result()

            except Exception as e:

                logger.error(
                    f"Case failed: {e}"
                )