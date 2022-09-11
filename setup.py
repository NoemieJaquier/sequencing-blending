from setuptools import setup, find_packages

# get description from readme file
with open('README.md', 'r') as f:
    long_description = f.read()

# setup
setup(
    name='SkillsSequencing',
    version='',
    description='',
    long_description = long_description,
    long_description_content_type="text/markdown",
    author='Noémie Jaquier, You Zhou, Julia Starke, Tamim Asfour',
    author_email='noemie.jaquier@kit.edu',
    maintainer=' ',
    maintainer_email='',
    license=' ',
    url=' ',
    platforms=['Linux Ubuntu'],
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
)
