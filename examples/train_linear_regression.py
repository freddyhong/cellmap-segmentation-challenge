import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
from upath import UPath
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy import ndimage
from skimage import filters
import torch
import seaborn as sns
from tqdm import tqdm
from cellmap_segmentation_challenge.utils import get_dataloader

def extract_features(image):
    if image.ndim == 3 and image.shape[0] > 1:
        img = image[0]
    else:
        img = image.squeeze()

    features = []

    # Gray Scale
    features.append(img.flatten())
    
    # Gaussian filters (reduced from 3 to 2)
    for sigma in [0.5, 1.0]:
        gaussian = filters.gaussian(img, sigma=sigma)
        features.append(gaussian.flatten())

    # Sobel
    sobel = filters.sobel(img)
    features.append(sobel.flatten())

    # Gradient Magnitude
    grad_x = ndimage.sobel(img, axis=1)
    grad_y = ndimage.sobel(img, axis=0)
    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
    features.append(gradient_mag.flatten())

    feature_matrix = np.stack(features, axis=0).T
    return feature_matrix
    
def get_batch_data(batch, data_loader):
    dataset = data_loader.dataset
    if hasattr(dataset, 'input_arrays'):
        input_keys = list(dataset.input_arrays.keys())
        target_keys = list(dataset.target_arrays.keys())
    elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'input_arrays'):
        input_keys = list(dataset.dataset.input_arrays.keys())
        target_keys = list(dataset.dataset.target_arrays.keys())
    else:
        input_keys = [k for k in batch.keys() if 'input' in k.lower()]
        target_keys = [k for k in batch.keys() if k not in input_keys]
        if not input_keys:
            input_keys = [list(batch.keys())[0]]
        if not target_keys:
            target_keys = [list(batch.keys())[-1]]
    
    if len(input_keys) > 1:
        inputs = torch.cat([batch[key] for key in input_keys], dim=1)
    else:
        inputs = batch[input_keys[0]]
    
    if len(target_keys) > 1:
        targets = torch.cat([batch[key] for key in target_keys], dim=1)
    else:
        targets = batch[target_keys[0]]
    
    return inputs, targets

def conversion(data_loader, feature_name, chunk_size=10):
    X_list = []
    y_list = []
    
    X_chunks = []
    y_chunks = []
    
    data_loader.refresh()
    total_batches = len(data_loader.loader)
    
    print(f"Processing all {total_batches} batches...")
    
    current_X = []
    current_y = []
    
    for batch_idx, batch in enumerate(tqdm(data_loader.loader)):
        inputs, targets = get_batch_data(batch, data_loader)
        
        for i in range(inputs.shape[0]):
            img = inputs[i].cpu().numpy()
            target = targets[i].cpu().numpy()
            
            features = extract_features(img)
            
            if target.shape[0] > 1:
                labels = np.argmax(target, axis=0).flatten()
            else:
                labels = (target.squeeze() > 0.5).astype(int).flatten()
            
            current_X.append(features)
            current_y.append(labels)
        
        # Every chunk_size batches, consolidate and clear memory
        if (batch_idx + 1) % chunk_size == 0 or batch_idx == total_batches - 1:
            if current_X:
                # Stack current chunk
                chunk_X = np.vstack(current_X)
                chunk_y = np.hstack(current_y)
                
                # Store chunk info instead of full data to save memory
                X_chunks.append(chunk_X)
                y_chunks.append(chunk_y)
                
                print(f"Processed chunk {len(X_chunks)}: {chunk_X.shape[0]:,} pixels")
                
                # Clear current lists
                current_X.clear()
                current_y.clear()
                
                # Force garbage collection
                import gc
                gc.collect()
    
    # Final consolidation
    print("Consolidating all chunks...")
    if X_chunks:
        x = np.vstack(X_chunks)
        y = np.hstack(y_chunks)
    else:
        x = np.empty((0, len(feature_name)))
        y = np.empty(0)
    
    # Clear chunk lists
    X_chunks.clear()
    y_chunks.clear()
    import gc
    gc.collect()
    
    print(f"Final dataset shape: {x.shape}")
    print(f"Memory usage: {x.nbytes / 1e9:.2f} GB")
    
    # Print class distribution
    unique_classes, counts = np.unique(y, return_counts=True)
    print("Class distribution:")
    class_names_full = ['background', 'endo', 'cell', 'lyso', 'mito', 'nuc']
    for cls, count in zip(unique_classes, counts):
        name = class_names_full[cls] if cls < len(class_names_full) else f"class_{cls}"
        percentage = (count / len(y)) * 100
        print(f"  {name} (class {cls}): {count:,} pixels ({percentage:.2f}%)")
    
    return x, y

def main():
    batch_size = 8 
    input_array_info = {
        "shape": (1, 128, 128),
        "scale": (8, 8, 8),
    }
    target_array_info = {
        "shape": (1, 128, 128),
        "scale": (8, 8, 8),
    }
    random_seed = 42
    classes = ["endo", "cell", "lyso", "mito", "nuc"]
    class_names = ['background'] + classes
    datasplit_path = "datasplit.csv" 

    # Reduced feature set
    feature_names = ['gray_scale', 'gauss_0.5', 'gauss_1.0', 'sobel_edges', 'gradient_mag']

    spatial_transforms = {
        "mirror": {"axes": {"x": 0.5, "y": 0.5}},
        "transpose": {"axes": ["x", "y"]},
    }

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    print("Setting up data loaders...")
    train_loader, val_loader = get_dataloader(
        datasplit_path=datasplit_path,
        classes=classes,
        batch_size=batch_size,
        input_array_info=input_array_info,
        target_array_info=target_array_info,
        spatial_transforms=spatial_transforms,
        iterations_per_epoch=100,  # Reduced from 500
        device="cpu",
        weighted_sampler=False,
    )

    print("\nExtracting training features...")
    X_train, y_train = conversion(
        train_loader, 
        class_names
    )
    
    print("\nExtracting validation features...")  
    X_val, y_val = conversion(
        val_loader, 
        class_names
    )

    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print("\nTraining logistic regression...")
    model = LogisticRegression(
        random_state=random_seed,
        max_iter=1000,  # Reduced iterations
        class_weight='balanced',
        solver='liblinear'
    )

    model.fit(X_train_scaled, y_train)
    print("Training completed!")

    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")

    print("\n" + "-"*30)
    print("FEATURE IMPORTANCE")
    print("-"*30)

    if hasattr(model, 'coef_'):
        if model.coef_.ndim > 1:
            importance = np.mean(np.abs(model.coef_), axis=0)
        else:
            importance = np.abs(model.coef_)
        
        indices = np.argsort(importance)[::-1]
        print("Top features:")
        for i in range(min(len(feature_names), len(indices))):
            idx = indices[i]
            print(f"  {feature_names[idx]}: {importance[idx]:.4f}")

    print("\n" + "-"*30)
    print("CONFUSION MATRIX")
    print("-"*30)
    cm = confusion_matrix(y_val, y_val_pred)
    print("Validation confusion matrix:")
    print(cm)

    # Save results
    plt.figure(figsize=(8, 6))
    available_classes = [class_names[i] for i in range(len(np.unique(y_val)))]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=available_classes,
                yticklabels=available_classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("Confusion matrix saved as 'confusion_matrix.png'")

    # Save model
    import pickle
    model_path = "logistic_regression_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model, 
            'scaler': scaler, 
            'classes': classes,
            'feature_names': feature_names
        }, f)
    print(f"Model saved as '{model_path}'")

if __name__ == "__main__":
    main()