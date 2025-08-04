# RAG-college
### Description

The RAG pipelining model takes PDFs downloaded from my universities LMS and then stores the data in vector databases which are then used for data retrieval and respective prompt structuring and final answer generation using the Gemma3:1b model using Ollama 

Here are outputs of the `RAG_pipline.py` , compared with the source material 

<img width="1499" height="652" alt="Terminal Output" src="https://github.com/user-attachments/assets/94160ff6-d483-4ab7-aae1-e3e95b13b358" />

_(Terminal Output)_

<img width="460" height="698" alt="PDF Source" src="https://github.com/user-attachments/assets/ab9c42f4-d7cb-45b0-a308-406a9082a8ef" />

_(Source Material)_

### Instructions without using LMS scrapper

1. Place PDF's you wish to use in `downloaded_pdfs`
2. Run `data_ingestion.py`
3. Run `rag_pipeline.py`

### First time use (downloding Requirements.txt)
Run 
`pip install -r requirements.txt`
