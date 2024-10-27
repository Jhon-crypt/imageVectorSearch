# Required installations:
# pip install torch torchvision
# pip install qdrant-client
# pip install pymilvus
# pip install Pillow
# pip install numpy
# pip install scikit-learn
# pip install tqdm

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
)
import time
from tqdm import tqdm


class ImageVectorSearch:
    def __init__(self):
        # Initialize the model (using ResNet50 pre-trained)
        self.model = models.resnet50(pretrained=True)
        self.model.eval()
        # Remove the last classification layer
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])

        # Define image transformations
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Initialize vector databases
        self.init_qdrant()
        self.init_milvus()

    def init_qdrant(self):
        self.qdrant = QdrantClient(":memory:")  # Use disk path for persistence
        self.qdrant.recreate_collection(
            collection_name="images",
            vectors_config=VectorParams(size=2048, distance=Distance.COSINE),
        )

    def init_milvus(self):
        connections.connect(alias="default", host="localhost", port="19530")

        # Define collection schema
        dim = 2048
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        schema = CollectionSchema(fields=fields, description="image vectors")

        # Create collection
        self.collection_name = "image_vectors"
        if utility.exists_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
        self.milvus_collection = Collection(self.collection_name, schema)

        # Create index
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        self.milvus_collection.create_index("vector", index_params)

    def extract_features(self, image_path):
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        image = image.unsqueeze(0)

        # Extract features
        with torch.no_grad():
            features = self.model(image)
        return features.squeeze().numpy()

    def add_images(self, image_folder):
        image_files = [
            f for f in os.listdir(image_folder) if f.endswith((".jpg", ".jpeg", ".png"))
        ]

        # Process images and store vectors
        qdrant_points = []
        milvus_vectors = []
        milvus_ids = []

        for idx, image_file in enumerate(tqdm(image_files)):
            image_path = os.path.join(image_folder, image_file)
            vector = self.extract_features(image_path)

            # Add to Qdrant
            qdrant_points.append(
                PointStruct(
                    id=idx, vector=vector.tolist(), payload={"file_name": image_file}
                )
            )

            # Add to Milvus
            milvus_vectors.append(vector.tolist())
            milvus_ids.append(idx)

        # Batch insert into Qdrant
        self.qdrant.upsert(collection_name="images", points=qdrant_points)

        # Batch insert into Milvus
        self.milvus_collection.insert([milvus_ids, milvus_vectors])
        self.milvus_collection.flush()

    def search_similar(self, query_image_path, top_k=5):
        # Extract query image features
        query_vector = self.extract_features(query_image_path)

        # Search in Qdrant
        qdrant_start = time.time()
        qdrant_results = self.qdrant.search(
            collection_name="images", query_vector=query_vector.tolist(), limit=top_k
        )
        qdrant_time = time.time() - qdrant_start

        # Search in Milvus
        milvus_start = time.time()
        self.milvus_collection.load()
        milvus_results = self.milvus_collection.search(
            data=[query_vector.tolist()],
            anns_field="vector",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=top_k,
        )
        milvus_time = time.time() - milvus_start

        return {
            "qdrant": {"results": qdrant_results, "time": qdrant_time},
            "milvus": {"results": milvus_results, "time": milvus_time},
        }


def run_performance_test(searcher, test_image_path, iterations=10):
    qdrant_times = []
    milvus_times = []

    for _ in range(iterations):
        results = searcher.search_similar(test_image_path)
        qdrant_times.append(results["qdrant"]["time"])
        milvus_times.append(results["milvus"]["time"])

    print("\nPerformance Test Results:")
    print(f"Qdrant Average Search Time: {np.mean(qdrant_times):.4f}s")
    print(f"Milvus Average Search Time: {np.mean(milvus_times):.4f}s")
    print(f"Qdrant Standard Deviation: {np.std(qdrant_times):.4f}s")
    print(f"Milvus Standard Deviation: {np.std(milvus_times):.4f}s")

# Example usage
if __name__ == "__main__":
    # Initialize the search system
    searcher = ImageVectorSearch()

    # Add images from a folder
    image_folder = "test_images"
    searcher.add_images(image_folder)

    # Search for similar images
    query_image = "images/i.jpeg"
    results = searcher.search_similar(query_image)

    # Run performance tests
    run_performance_test(searcher, query_image)
