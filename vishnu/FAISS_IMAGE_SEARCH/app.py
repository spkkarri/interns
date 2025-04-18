%pip install streamlit
torch
torchvision
faiss-cpu
pillow
numpy
scikit-learn
import streamlit as st
import os
import time
import faiss
import torch
import numpy as np
from torchvision import models, transforms
from sklearn.decomposition import PCA
from PIL import Image, ImageFile, ExifTags

# Allow truncated images to load
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Feature Extractor Class
class FeatureExtractor:
    def __init__(self, pca_components=128):
        self.model = models.resnet18(pretrained=True)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.pca = PCA(n_components=pca_components)
        self.pca_trained = False

    def extract(self, image):
        try:
            image = image.convert('RGB')
            image = self.transform(image).unsqueeze(0)
            with torch.no_grad():
                feature = self.model(image).squeeze().numpy()
            return feature / np.linalg.norm(feature)
        except Exception as e:
            st.error(f"Error processing image: {e}")
            return None

    def apply_pca(self, features):
        if not self.pca_trained:
            self.pca.fit(features)
            self.pca_trained = True
        return self.pca.transform(features)

# Metadata functions
def extract_metadata(image):
    exif_data = image.getexif()
    metadata = {}
    if exif_data:
        for tag, value in exif_data.items():
            tag_name = ExifTags.TAGS.get(tag, tag)
            if tag_name in ['Rating']:
                try:
                    metadata[tag_name] = float(value)
                except ValueError:
                    pass
    return metadata

def scale_metadata(metadata, size=5):
    rating = metadata.get('Rating', 0.0)
    return np.full(size, rating / 5.0, dtype=np.float32)

# Index dataset function for uploaded files
@st.cache_resource
def index_dataset(uploaded_files):
    extractor = FeatureExtractor(pca_components=128)
    file_paths = []
    features = []
    metadata_vectors = []
    
    with st.spinner("Indexing uploaded dataset images..."):
        for uploaded_file in uploaded_files:
            if uploaded_file.name.endswith(('.png', '.jpg', '.jpeg')):
                image = Image.open(uploaded_file)
                feature = extractor.extract(image)
                metadata = extract_metadata(image)
                if feature is not None:
                    file_paths.append(uploaded_file.name)
                    metadata_vector = scale_metadata(metadata)
                    features.append(feature)
                    metadata_vectors.append(metadata_vector)
    
    if not features:
        st.error("No valid images found in uploaded files!")
        return None, None, None
    
    features = np.array(features, dtype='float32')
    metadata_vectors = np.array(metadata_vectors, dtype='float32')
    reduced_features = extractor.apply_pca(features)
    combined_vectors = np.hstack((reduced_features, metadata_vectors))
    d = combined_vectors.shape[1]
    
    hnsw_index = faiss.IndexHNSWFlat(d, 32)
    hnsw_index.add(combined_vectors)
    
    return hnsw_index, file_paths, extractor

# Search function (modified to accept k as a parameter)
def search_similar_images(query_image, index, file_paths, extractor, stored_metadata, k):
    query_feature = extractor.extract(query_image)
    if query_feature is None:
        return [], {}
    
    query_feature = extractor.apply_pca(query_feature.reshape(1, -1))[0]
    metadata_vector = scale_metadata(stored_metadata)
    query_vector = np.hstack((query_feature, metadata_vector)).reshape(1, -1)
    
    distances, indices = index.search(query_vector, k)
    results = [file_paths[idx] for idx in indices[0]]
    return results, stored_metadata

# Streamlit UI
def main():
    st.title("Similar Image Search")
    st.write("Upload multiple index images and a query image to find similar images")

    # Upload multiple index files
    st.subheader("Upload Index Images (Dataset) press Ctrl+a to select all the images")
    uploaded_index_files = st.file_uploader(
        "Choose multiple images for indexing...",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )

    # Initialize index when files are uploaded
    if uploaded_index_files:
        if 'index' not in st.session_state or st.session_state.get('index_files') != [f.name for f in uploaded_index_files]:
            st.session_state.index, st.session_state.file_paths, st.session_state.extractor = index_dataset(uploaded_index_files)
            st.session_state.index_files = [f.name for f in uploaded_index_files]
            if st.session_state.index is not None:
                st.success(f"Dataset indexed successfully with {len(uploaded_index_files)} images!")
            else:
                st.error("Failed to index dataset. Please check your uploaded files.")

    # Upload query image
    st.subheader("Upload Query Image")
    uploaded_query_file = st.file_uploader(
        "Choose a query image...",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=False
    )

    if uploaded_query_file is not None and 'index' in st.session_state and st.session_state.index is not None:
        # Display uploaded query image
        query_image = Image.open(uploaded_query_file)
        st.image(query_image, caption='Uploaded Query Image', use_container_width=True)
        
        # User input for number of similar images (top-k)
        k = st.number_input(
            "Number of similar images to retrieve",
            min_value=1,
            max_value=len(st.session_state.file_paths),  # Limit to dataset size
            value=3,  # Default value
            step=1
        )
        
        # Use the first index image's metadata as stored metadata (for simplicity)
        if uploaded_index_files:
            sample_metadata_image = Image.open(uploaded_index_files[0])
            stored_metadata = extract_metadata(sample_metadata_image)
        
        # Search button
        if st.button("Search Similar Images"):
            with st.spinner("Searching for similar images..."):
                similar_images, metadata_used = search_similar_images(
                    query_image,
                    st.session_state.index,
                    st.session_state.file_paths,
                    st.session_state.extractor,
                    stored_metadata,
                    k  # Pass user-defined k
                )
            
            # Display results
            st.subheader("Similar Images Found:")
            if similar_images:
                # Dynamically adjust columns based on k, max 3 per row
                num_cols = min(3, k)
                cols = st.columns(num_cols)
                for i, img_name in enumerate(similar_images):
                    matching_file = next(f for f in uploaded_index_files if f.name == img_name)
                    img = Image.open(matching_file)
                    cols[i % num_cols].image(img, caption=img_name, use_container_width=True)
                
                st.write("Metadata used for search:", metadata_used)
            else:
                st.warning("No similar images found.")

if __name__ == "__main__":
    main()