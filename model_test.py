from collections import defaultdict
from utils import apply_high_pass_filter
from utils import smash_n_reconstruct
kernels = apply_high_pass_filter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageClassificationModel(kernels).to(device)

def test(model, test_loader, device):
    # Load the best model
    checkpoint = torch.load("/home/kosta/code/School/SentryAI/pth/best_model_newPatching.pth")
    model.load_state_dict(checkpoint['model_state'])
    
    model.eval()
    total_test, correct_test = 0, 0
    test_accuracy_per_model = defaultdict(lambda: {'correct': 0, 'total': 0})

    with torch.no_grad():
        for batch in test_loader:
            rich, poor, labels, model_names = batch  # Assuming you have model_names
            rich = rich.to(device)
            poor = poor.to(device)
            labels = labels.to(device)

            outputs = model(rich, poor)
            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

            # Collect stats per model just like validation phase
            for model_name, pred, true in zip(model_names, predicted, labels):
                test_accuracy_per_model[model_name]['total'] += 1
                if pred == true:
                    test_accuracy_per_model[model_name]['correct'] += 1

    test_accuracy = correct_test / total_test
    print(f'Test Accuracy: {test_accuracy:.4f}')

    # Print per model accuracy
    print("-------------------------------------------------------------------------")
    print("Test Accuracy per model:")
    for model_name, stats in test_accuracy_per_model.items():
        model_accuracy = stats['correct'] / stats['total']
        print(f"Test Accuracy for model {model_name}: {model_accuracy:.4f}")

test(model, test_loader, device)
from PIL import Image
import torchvision.transforms as transforms
import torch

# Load the model
checkpoint = torch.load("/home/kosta/code/School/SentryAI/pth/best_model_newPatching.pth")
model.load_state_dict(checkpoint['model_state'])
model.to(device)
model.eval()

# Define the image path
img_path = '/mnt/c/Users/kosta/Downloads/Screenshot 2024-04-28 002208.png'

# Load the image
rich, poor  = smash_n_reconstruct(Image.open(img_path).convert('RGB'))

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
rich_tensor = transform(rich)
poor_tensor = transform(poor)

# Predict
with torch.no_grad():
    output = model(rich_tensor.unsqueeze(0).to(device), poor_tensor.unsqueeze(0).to(device))
    _, predicted = torch.max(output, 1)

# Print the predicted class
print("Predicted class:", predicted.item())

