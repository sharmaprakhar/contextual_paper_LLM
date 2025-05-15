# contextual_paper_LLM
This is an implementation of contextual retrieval for academic papers. The current version sets up a 5G network security expert LLM and a db index geared towards efficient information extraction from 4G/5G/LTE academic papers.

# Environment
This code was tested on python 3.11. Currently only GPU (cuda) environments supported. 

# Install dependencies
conda create -n contextual_paper_retrieval python=3.11
<br>
conda activate contextual_paper_retrieval
<br>
pip install -r requirements.txt