""" See:
https://github.com/leo-beck/AutoEMA
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="AutoEMA",
    version="0.0.3",
    description="A fully automated EMA (Experimental Modal Analysis) using Bayesian optimization",
    url="https://github.com/leo-beck/AutoEMA",
    author="Leopold Beck",
    author_email="l.beck@tum.de",
    keywords="EMA, modal, analysis, automated, mechanics, bayes, bayesian, optimization",
    package_dir={"": "src"},  # Optional
    packages=find_packages(where="src"),
    python_requires=">=3.7, <4",
    install_requires=['matplotlib>=3.5.3',
                      'numpy>=1.23.2',
                      'scikit-learn>=1.1.2',
                      'scipy==1.7.2',
                      'bayesian-optimization>=1.2.0',
                      'sdypy-EMA>=0.24']
)
