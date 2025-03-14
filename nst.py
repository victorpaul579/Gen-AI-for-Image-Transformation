import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
import os
from pathlib import Path

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        # The first number x in convx_y gets added by 1 after it has gone
        # through a maxpool, and the second y if we have several conv layers
        # in between a max pool. These strings (0, 5, 10, ..) then correspond
        # to conv1_1, conv2_1, conv3_1, conv4_1, conv5_1 mentioned in NST paper
        self.chosen_features = ["0", "5", "10", "19", "28"]
        # We don't need to run anything further than conv5_1 (the 28th module in vgg)
        # Since remember, we dont actually care about the output of VGG: the only thing
        # that is modified is the generated image (i.e, the input).
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        # Store relevant features
        features = []
        # Go through each layer in model, if the layer is in the chosen_features,
        # store it in features. At the end we'll just return all the activations
        # for the specific layers we have in chosen_features
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features

def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)

def apply_style_transfer(original_img, style_img, save_path, total_steps=2000):
    """
    Apply neural style transfer to combine content of original_img with style of style_img
    
    # The process follows the original NST paper:
    # 1. Initialize generated image
    # 2. Forward pass through VGG19
    # 3. Calculate content and style losses
    # 4. Update generated image via backpropagation
    """
    # initialized generated as clone of original image.
    # Clone seemed to work better than white noise initialization
    generated = original_img.clone().requires_grad_(True)
    
    # Hyperparameters
    learning_rate = 0.001
    alpha = 1  # Content loss weight
    beta = 0.01  # Style loss weight
    optimizer = optim.Adam([generated], lr=learning_rate)
    
    for step in range(total_steps):
        # Obtain the convolution features in specifically chosen layers
        generated_features = model(generated)
        original_img_features = model(original_img)
        style_features = model(style_img)
        
        # Loss is 0 initially
        style_loss = original_loss = 0
        
        # iterate through all the features for the chosen layers
        for gen_feature, orig_feature, style_feature in zip(
            generated_features, original_img_features, style_features
        ):
            # batch_size will just be 1
            batch_size, channel, height, width = gen_feature.shape
            # Calculate content loss as MSE between features
            original_loss += torch.mean((gen_feature - orig_feature) ** 2)
            
            # Compute Gram Matrix of generated
            G = gen_feature.view(channel, height * width).mm(
                gen_feature.view(channel, height * width).t()
            )
            # Compute Gram Matrix of Style
            A = style_feature.view(channel, height * width).mm(
                style_feature.view(channel, height * width).t()
            )
            style_loss += torch.mean((G - A) ** 2)
        
        total_loss = alpha * original_loss + beta * style_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if step % 200 == 0:
            print(f"Step [{step}/{total_steps}], Total loss: {total_loss.item():.4f}")
            # Save intermediate result
            save_image(generated, save_path)
    
    # Save final result
    save_image(generated, save_path)

def main():
    """
    Main function to process multiple style transfers
    - Reads content image from saved/00003.png
    - Applies each style from the styles folder
    - Saves results in the generated folder
    """
    # Create output directory if it doesn't exist
    output_dir = Path("generated")
    output_dir.mkdir(exist_ok=True)
    
    # Load content image
    content_image_path = "saved/00003.png"
    print(f"\nLoading content image from: {content_image_path}")
    original_img = load_image(content_image_path)
    
    # Process each style image in the styles folder
    styles_dir = Path("styles")
    style_images = list(styles_dir.glob("*.jpg"))
    print(f"Found {len(style_images)} style images to process")
    
    for i, style_path in enumerate(style_images, 1):
        print(f"\nProcessing style {i}/{len(style_images)}: {style_path}")
        
        # Load style image
        style_img = load_image(style_path)
        
        # Create output filename based on content and style names
        output_filename = f"generated_{style_path.stem}.png"
        output_path = output_dir / output_filename
        
        # Apply style transfer
        print(f"Applying style transfer... This may take a while.")
        apply_style_transfer(original_img, style_img, output_path)
        print(f"Style transfer complete! Generated image saved as: {output_path}")

if __name__ == "__main__":
    # Set device and image size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    imsize = 356
    
    # Here we may want to use the Normalization constants used in the original
    # VGG network (to get similar values net was originally trained on), but
    # I found it didn't matter too much so I didn't end of using it. If you
    # use it make sure to normalize back so the images don't look weird.
    loader = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Initialize model
    model = VGG().to(device).eval()
    
    main()