import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import traceback

from models import NeuralCA, SimpleCNN, prepare_nca_input

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
st.sidebar.write(f"Using device: {device}")

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

@st.cache_resource
def load_models():
    try:
        nca_model = NeuralCA(channel_n=16).to(device)
        nca_model.load_state_dict(torch.load('models/nca_model.pth', map_location=device))
        nca_model.eval()
        
        cnn_model = SimpleCNN().to(device)
        cnn_model.load_state_dict(torch.load('models/cnn_model.pth', map_location=device))
        cnn_model.eval()
        
        return nca_model, cnn_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def create_masked_image(image, mask_ratio=0.4):
    if image.device != torch.device('cpu'):
        image_cpu = image.cpu()
    else:
        image_cpu = image
        
    mask = torch.rand_like(image_cpu) > mask_ratio
    masked_img = image_cpu * mask.float()
    return masked_img.to(device), mask.float().to(device)

def nca_step_by_step(nca_model, initial_state, steps=40, mask=None):
    device = next(nca_model.parameters()).device
    x = initial_state.clone()
    
    with torch.no_grad():
        for step in range(steps):
            perception = nca_model.perception(x)
            dx = nca_model.update(perception)
            
            fire_mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=device) < nca_model.fire_rate).float()
            dx = dx * fire_mask
            
            x = x + dx
            x = torch.clamp(x, 0, 1)
            
            if mask is not None:
                x[:, 0:1] = x[:, 0:1] * (1 - mask.unsqueeze(1)) + initial_state[:, 0:1] * mask.unsqueeze(1)
            
            yield x.cpu(), step + 1

def process_image_with_animation(nca_model, cnn_model, image_tensor, mask_ratio=0.4, steps=40):
    with torch.no_grad():
        if image_tensor.device != device:
            image_tensor = image_tensor.to(device)
        
        masked_img, mask = create_masked_image(image_tensor, mask_ratio)
        
        if len(masked_img.shape) == 3:
            masked_img = masked_img.unsqueeze(1)
        
        state = prepare_nca_input(masked_img.squeeze(1), nca_model.channel_n)
        state = state.to(device)
        state[:, 0:1] = masked_img
        
        final_state = None
        for reconstructed_state, step in nca_step_by_step(nca_model, state, steps, mask):
            final_state = reconstructed_state
            yield reconstructed_state, step, False
        
        reconstructed_final = final_state.to(device)
        predictions = cnn_model(reconstructed_final[:, 0:1])
        pred_probs = F.softmax(predictions, dim=1)
        pred_label = predictions.argmax(dim=1).item()
        confidence = pred_probs[0, pred_label].item()
        
        yield final_state, steps, True, pred_label, confidence

def tensor_to_display(image_tensor):
    if image_tensor.device != torch.device('cpu'):
        image_tensor = image_tensor.cpu()
    
    if len(image_tensor.shape) == 4:
        if image_tensor.shape[1] == 1:
            return image_tensor.squeeze(1).squeeze(0).numpy()
        else:
            return image_tensor[:, 0:1].squeeze(1).squeeze(0).numpy()
    elif len(image_tensor.shape) == 3:
        if image_tensor.shape[0] == 1:
            return image_tensor.squeeze(0).numpy()
        elif image_tensor.shape[2] == 1:
            return image_tensor.squeeze(2).numpy()
        else:
            return image_tensor[0].numpy()
    elif len(image_tensor.shape) == 2:
        return image_tensor.numpy()
    else:
        return image_tensor.squeeze().numpy()
    
def main():
    st.set_page_config(page_title="NCA Image Reconstruction & Classification", layout="wide")
    
    st.title("🧠 Neural Cellular Automata - Real-time Image Reconstruction")
    st.markdown("""
    This app demonstrates **real-time NCA reconstruction** of masked Fashion-MNIST images:
    - Watch the Neural Cellular Automata **gradually reconstruct** the image
    - See the **step-by-step evolution** from masked to reconstructed
    - **CNN Classification** of the final reconstructed image
    """)
    
    nca_model, cnn_model = load_models()
    
    if nca_model is None or cnn_model is None:
        st.error("❌ Failed to load models. Please check if model files exist in the 'models' directory.")
        st.info("Required files: `nca_model.pth` and `cnn_model.pth`")
        return
    else:
        st.success("✅ Models loaded successfully!")
    
    st.sidebar.header("Controls")
    
    option = st.sidebar.radio(
        "Choose image source:",
        ["Random Fashion-MNIST", "Upload your own image"]
    )
    
    mask_ratio = st.sidebar.slider(
        "Masking Ratio", 
        min_value=0.1, 
        max_value=0.8, 
        value=0.4,
        help="Percentage of the image to mask out"
    )
    
    nca_steps = st.sidebar.slider(
        "NCA Steps", 
        min_value=10, 
        max_value=100, 
        value=40,
        help="Number of NCA evolution steps"
    )
    
    animation_speed = st.sidebar.slider(
        "Animation Speed (ms per step)", 
        min_value=50, 
        max_value=500, 
        value=100,
        help="Speed of the real-time reconstruction"
    )
    
    @st.cache_data
    def load_fashion_mnist():
        try:
            from torchvision import datasets, transforms
            test_dataset = datasets.FashionMNIST(
                root='./data', train=False, download=True, transform=transforms.ToTensor()
            )
            return test_dataset
        except Exception as e:
            st.error(f"Error loading Fashion-MNIST: {e}")
            return None
    
    if option == "Random Fashion-MNIST":
        test_dataset = load_fashion_mnist()
        
        if test_dataset is None:
            return
            
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("🎲 Random Image"):
                st.session_state.random_idx = np.random.randint(len(test_dataset))
                if 'animation_done' in st.session_state:
                    del st.session_state.animation_done
        with col2:
            if st.button("🔄 Same Image"):
                if 'random_idx' not in st.session_state:
                    st.session_state.random_idx = np.random.randint(len(test_dataset))
        
        if 'random_idx' not in st.session_state:
            st.session_state.random_idx = np.random.randint(len(test_dataset))
        
        idx = st.session_state.random_idx
        image, true_label = test_dataset[idx]
        
        st.sidebar.info(f"Image Index: {idx} | True Label: {class_names[true_label]}")
        
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload a grayscale image (28x28)", 
            type=['png', 'jpg', 'jpeg']
        )
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert('L')
                image = image.resize((28, 28))
                image_array = np.array(image) / 255.0
                image_tensor = torch.tensor(image_array, dtype=torch.float32)
                true_label = -1
                
                st.session_state.uploaded_image = image_tensor
                st.session_state.uploaded_true_label = true_label
                
            except Exception as e:
                st.error(f"Error processing uploaded image: {e}")
                return
        else:
            st.info("👆 Please upload an image to get started")
            return
    
    if st.sidebar.button("🚀 Start Real-time Reconstruction"):
        if option == "Random Fashion-MNIST":
            image_tensor = image
            current_true_label = true_label
        else:
            if 'uploaded_image' not in st.session_state:
                st.warning("Please upload an image first")
                return
            image_tensor = st.session_state.uploaded_image
            current_true_label = st.session_state.uploaded_true_label
        
        masked_img, mask = create_masked_image(image_tensor, mask_ratio)

        original_np = tensor_to_display(image_tensor.cpu())
        masked_np = tensor_to_display(masked_img.cpu())
                
        progress_bar = st.progress(0)
        status_text = st.empty()
        image_placeholder = st.empty()
        results_placeholder = st.empty()
        
        try:
            for reconstructed_state, step, is_final, *classification_info in process_image_with_animation(
                nca_model, cnn_model, image_tensor, mask_ratio, nca_steps
            ):
                progress = step / nca_steps
                progress_bar.progress(progress)
                
                reconstructed_np = tensor_to_display(reconstructed_state[:, 0:1])
                
                if is_final:
                    status_text.success(f"✅ Reconstruction Complete! Step {step}/{nca_steps}")
                else:
                    status_text.info(f"🔄 Reconstructing... Step {step}/{nca_steps}")
                
                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                
                axes[0].imshow(original_np, cmap='gray', vmin=0, vmax=1)
                axes[0].set_title('Original Image', fontsize=14, weight='bold')
                axes[0].axis('off')
                
                axes[1].imshow(masked_np, cmap='gray', vmin=0, vmax=1)
                axes[1].set_title(f'Masked Input\n({mask_ratio*100:.1f}% missing)', fontsize=14, weight='bold')
                axes[1].axis('off')
                
                axes[2].imshow(reconstructed_np, cmap='gray', vmin=0, vmax=1)
                axes[2].set_title(f'NCA Reconstruction\nStep {step}/{nca_steps}', fontsize=14, weight='bold')
                axes[2].axis('off')
                
                if is_final:
                    pred_label, confidence = classification_info
                    if option == "Random Fashion-MNIST":
                        is_correct = (current_true_label == pred_label)
                        result_color = 'green' if is_correct else 'red'
                        result_symbol = '✓' if is_correct else '✗'
                        
                        axes[3].text(0.5, 0.7, 'Predicted:', ha='center', va='center', fontsize=12)
                        axes[3].text(0.5, 0.55, f'{class_names[pred_label]}', 
                                   ha='center', va='center', fontsize=16, weight='bold', color=result_color)
                        axes[3].text(0.5, 0.4, f'Confidence: {confidence*100:.1f}%',
                                   ha='center', va='center', fontsize=11)
                        axes[3].text(0.5, 0.25, 'True Label:', ha='center', va='center', fontsize=12)
                        axes[3].text(0.5, 0.1, f'{class_names[current_true_label]}',
                                   ha='center', va='center', fontsize=14, style='italic')
                        axes[3].text(0.5, 0.9, f'{result_symbol}', 
                                   ha='center', va='center', fontsize=40, weight='bold', color=result_color)
                    else:
                        axes[3].text(0.5, 0.6, 'Predicted:', ha='center', va='center', fontsize=12)
                        axes[3].text(0.5, 0.45, f'{class_names[pred_label]}', 
                                   ha='center', va='center', fontsize=16, weight='bold', color='blue')
                        axes[3].text(0.5, 0.3, f'Confidence: {confidence*100:.1f}%',
                                   ha='center', va='center', fontsize=11)
                        axes[3].text(0.5, 0.15, 'True Label: Unknown',
                                   ha='center', va='center', fontsize=12, style='italic')
                else:
                    axes[3].text(0.5, 0.6, 'Reconstruction in Progress', 
                               ha='center', va='center', fontsize=14, weight='bold')
                    axes[3].text(0.5, 0.4, f'Step: {step}/{nca_steps}', 
                               ha='center', va='center', fontsize=12)
                    axes[3].text(0.5, 0.2, 'Classification pending...', 
                               ha='center', va='center', fontsize=10, style='italic')
                
                axes[3].set_xlim(0, 1)
                axes[3].set_ylim(0, 1)
                axes[3].axis('off')
                
                plt.tight_layout()
                image_placeholder.pyplot(fig)
                plt.close(fig)
                
                if not is_final:
                    time.sleep(animation_speed / 1000.0)
                else:
                    st.session_state.animation_done = True
                    pred_label, confidence = classification_info
                    
                    results_placeholder.subheader("📊 Final Results Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Predicted Class", class_names[pred_label])
                    with col2:
                        st.metric("Confidence", f"{confidence*100:.1f}%")
                    with col3:
                        if option == "Random Fashion-MNIST":
                            is_correct = (current_true_label == pred_label)
                            status = "Correct ✓" if is_correct else "Incorrect ✗"
                            st.metric("Accuracy", status, 
                                    delta="Correct" if is_correct else "Incorrect",
                                    delta_color="normal" if is_correct else "inverse")
                    
                    results_placeholder.subheader("📈 Classification Probabilities")
                    with torch.no_grad():
                        reconstructed_final_for_probs = reconstructed_state.to(device)
                        if len(reconstructed_final_for_probs.shape) == 4:
                            reconstructed_final_for_probs = reconstructed_final_for_probs[:, 0:1]
                        
                        predictions = cnn_model(reconstructed_final_for_probs)
                        all_probs = F.softmax(predictions, dim=1).cpu().numpy()[0]
                    
                    fig_prob, ax = plt.subplots(figsize=(10, 6))
                    y_pos = np.arange(len(class_names))
                    colors = ['lightblue' if i != pred_label else 'red' for i in range(len(class_names))]
                    
                    bars = ax.barh(y_pos, all_probs, color=colors, alpha=0.7)
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(class_names)
                    ax.invert_yaxis()
                    ax.set_xlabel('Probability')
                    ax.set_title('Final Classification Probabilities')
                    
                    for i, (bar, prob) in enumerate(zip(bars, all_probs)):
                        width = bar.get_width()
                        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                               f'{prob:.3f}', ha='left', va='center')
                    
                    plt.tight_layout()
                    results_placeholder.pyplot(fig_prob)
                    plt.close(fig_prob)
                
        except Exception as e:
            st.error(f"Error during processing: {e}")
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()