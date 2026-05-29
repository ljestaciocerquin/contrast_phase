from guidereg.execution.sequential import (
    run_sequential
)

from guidereg.execution.multiprocessing_backend import (
    run_multiprocessing
)


def dispatch_execution(
    cases,
    pipeline_function,
    config,
    logger,
    backend="sequential",
    num_workers=4
):

    if backend == "sequential":

        run_sequential(
            cases,
            pipeline_function,
            config,
            logger
        )

    elif backend == "multiprocessing":

        run_multiprocessing(
            cases,
            pipeline_function,
            config,
            logger,
            num_workers
        )

    else:

        raise ValueError(
            f"Unknown backend: "
            f"{backend}"
        )