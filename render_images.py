import torch
from nerfstudio.models.nerfacto import NerfactoModel

# Load your trained Nerfacto model
model = NerfactoModel.load_from_checkpoint(
    "/juno/u/jingyuny/projects/p_mobilization/outputs/splat_data/nerfacto/2024-11-20_170619/nerfstudio_models/step-000008000.ckpt"
)
model.eval()  # Set to evaluation mode
for param in model.parameters():
    param.requires_grad = False  # Freeze model parameters

# Define your input extrinsics and intrinsics
camera_extrinsics = torch.eye(4, requires_grad=True)  # Example: identity matrix
camera_intrinsics = torch.tensor(
    [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], requires_grad=True
)

# Render an image using the Nerfacto model
rendered_image = model.render(camera_extrinsics, camera_intrinsics)

# Define a loss (e.g., mean pixel intensity for demonstration)
target_image = torch.zeros_like(rendered_image)  # Replace with your actual target image
loss = torch.nn.functional.mse_loss(rendered_image, target_image)

# Backpropagate to compute gradients w.r.t. extrinsics
loss.backward()

# Check gradients
print(camera_extrinsics.grad)  # Gradients w.r.t. extrinsics
print(camera_intrinsics.grad)  # Gradients w.r.t. intrinsics

# Optionally, update extrinsics using an optimizer (if needed)
optimizer = torch.optim.Adam([camera_extrinsics], lr=0.01)
optimizer.step()
