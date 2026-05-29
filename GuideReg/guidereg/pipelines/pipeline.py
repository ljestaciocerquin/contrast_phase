from guidereg.datasets.loader import load_dataset
from guidereg.datasets.registration_dataset import build_registration_cases
from guidereg.pipelines.registration_pipeline    import run_registration_pipeline
from guidereg.execution.dispatcher          import dispatch_execution

def run_pipeline(
    config,
    logger
):

    logger.info(
        "Loading dataset"
    )
    dataframe = load_dataset(config["dataset"]["input_csv"])
    logger.info(
        f"Loaded {len(dataframe)} rows"
    )

    # ======================================
    # BUILD CASES
    # ======================================
    cases = build_registration_cases(
        dataframe,
        config
    )

    logger.info(
        f"Built {len(cases)} "
        f"registration cases"
    )

    # ======================================
    # EXECUTE CASES
    # ======================================
    dispatch_execution(
        cases               =   cases,
        pipeline_function   =   run_registration_pipeline,
        config              =   config,
        logger              =   logger,
        backend             =   config["execution"]["backend"],
        num_workers         =   config["execution"]["num_workers"]
    )