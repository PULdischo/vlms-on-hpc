https://researchcomputing.princeton.edu/support/knowledge-base/hugging-face

The HPC node does not have access to the Internet, so you need to download all model and image files in advance on the login node. 

To do this: 
`python fetch.py model`
then 
`python fetch.py images <IIIF manifest URL>`

