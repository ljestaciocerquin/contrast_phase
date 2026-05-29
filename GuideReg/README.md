## Installing requirements

- Virtual environment

        $ conda create -n guidereg python=3.10
        $ conda activate guidereg
        $ conda install -c conda-forge itk-elastix

## Installing packages (inside the virtual environment)

        $ git clone https://github.com/ljestaciocerquin/GuideReg.git
        $ cd Guidereg
        $ pip install -e .


# Execute
        $ cd guidereg
        $ guidereg pipeline --config configs/examples/liver_ct.yaml

# Execute only for segmentation
        $ cd guidereg
        $ guidereg propagate --config configs/examples/propagate_tumor.yaml