## Create a python environment
`python -m venv env_name
`
## Activate the virtual environment:
### On windows:
```bash
.\env_name\Scripts\activate
```

### On macOS:
```
source env_name/bin/activate
```

## Install all required packages
```bash
pip install torch torchvision
pip install qdrant-client
pip install pymilvus
pip install Pillow
pip install numpy
pip install scikit-learn
pip install tqdm
```
## Install Vector Databases:

### Start Milvus
```python docker pull milvusdb/milvus:latest
docker run -d --name milvus_standalone -p 19530:19530 -p 19531:19531 milvusdb/milvus:latest
```

### Start Qdrant

```docker pull qdrant/qdrant
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

### Then run 
```python
python main.py
```