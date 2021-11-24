conda create --name pytest pandas numpy pytest scipy matplotlib seaborn

conda activate pytest

conda env export > environment.yml
