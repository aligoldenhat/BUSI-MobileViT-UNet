# Breast Ultrasound Tumor Segmentation using MobileViT-UNet
This model combines a pre-trained **MobileViTV2** vision transformer as the **Encoder** with a **U-Net style Decoder** for segmentation.

i use [Breast Ultrasound Images Dataset (BUSI)](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) hosted on Kaggle.

**Inspired by:** [PMC11429406](https://pmc.ncbi.nlm.nih.gov/articles/PMC11429406)

## Model Architecture: MobileViT-UNet
**1. Encoder:**

*   **Base Model:** Uses a pre-trained **MobileViTV2** model (`apple/mobilevitv2-1.0-imagenet1k-256`)
*   **Feature Extraction:** The MobileViTV2 encoder processes the input image (resized to 224x224) through its layers, generating hierarchical feature representations.
*   **Hidden States:** The MobileViTV2 model is configured with `output_hidden_states=True`. This allows access to the output feature maps from multiple stages of the encoder, which are crucial for the U-Net skip connections.
*   **Skip Connection Features:** The hidden states captured correspond to feature maps with the following approximate channel dimensions (based on the specific MobileViTV2 variant used):
    *   `hs[0]`: ~64 channels (Earliest, highest resolution features used)
    *   `hs[1]`: ~128 channels
    *   `hs[2]`: ~256 channels
    *   `hs[3]`: ~384 channels
*   **Bottleneck:** The final output of the MobileViTV2 encoder (`hs[4]`) serves as the bottleneck layer, having 512 channels.

**2. Decoder (U-Net Style Upsampling Path):**

*   The decoder consists of four custom `UNetDecoderBlock` stages designed to progressively upsample the feature maps while integrating high-resolution features from the encoder via skip connections.
*   **`UNetDecoderBlock` Structure:** Each block performs the following operations:
    1.  **Upsampling:** Takes the feature map from the previous decoder stage (or the bottleneck for the first stage) and doubles its spatial dimensions using a `nn.ConvTranspose2d` layer (kernel size 2, stride 2). The number of channels is typically halved during this step (e.g., 512 -> 256 in the first block).
    2.  **Skip Connection Concatenation:** The upsampled feature map is concatenated channel-wise with the corresponding hidden state (feature map) from the MobileViTV2 encoder. Padding (`F.pad`) is applied if necessary to ensure spatial dimensions match before concatenation.
    3.  **Convolutional Refinement:** The concatenated feature map is passed through a sequence of two 3x3 `nn.Conv2d` layers, each followed by a `nn.ReLU` activation function. This refines the features and adjusts the channel dimension for the next stage.

**3. Segmentation Head**
