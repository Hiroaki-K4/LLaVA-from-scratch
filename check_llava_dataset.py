import textwrap  # For text wrapping

import matplotlib.pyplot as plt

from llava_image_downloader import LLaVAImageDownloader


def create_llava_streaming_viewer(
    dataset_name: str = "liuhaotian/LLaVA-Instruct-150K", max_samples: int = 5
):
    """
    Display LLaVA dataset images and conversations with streaming support

    Args:
        dataset_name: HuggingFace dataset name
        max_samples: Number of samples to display
    """
    print("=== LLaVA-Instruct-150K Streaming Viewer ===\n")

    # Initialize image downloader
    downloader = LLaVAImageDownloader()

    try:
        from datasets import load_dataset

        print(f"Loading dataset: {dataset_name}")
        # Load dataset with streaming enabled to avoid downloading entire dataset
        dataset = load_dataset(dataset_name, split="train", streaming=True)

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please install datasets: pip install datasets")
        return

    print(f"Starting streaming from HuggingFace dataset...")

    success_count = 0
    sample_count = 0

    try:
        # Stream data from HuggingFace dataset
        for i, item in enumerate(dataset):
            if i >= max_samples:
                break

            sample_count = i + 1

            print(f"\n{'='*60}")
            print(f"Sample {sample_count}/{max_samples}")
            print(f"{'='*60}")

            # Extract data
            sample_id = item.get("id", f"sample_{i}")
            image_filename = item.get("image", "")
            conversations = item.get("conversations", [])

            print(f"ID: {sample_id}")
            print(f"Image file: {image_filename}")

            # Display conversation data
            print(f"\n📝 Conversations:")
            for j, conv in enumerate(conversations):
                role = conv.get("from", "unknown")
                content = conv.get("value", "").replace("<image>", "[image]")

                print(
                    f"  {j+1}. {role}: {content[:200]}..."
                    if len(content) > 200
                    else f"  {j+1}. {role}: {content}"
                )

            # Download and display image (without saving)
            if image_filename:
                print(f"\n🖼️  Processing image...")

                image = downloader.download_image_by_filename(image_filename)

                if image:
                    # Display image
                    plt.figure(figsize=(12, 8))

                    # Image display (left side)
                    plt.subplot(1, 2, 1)
                    plt.imshow(image)
                    plt.axis("off")
                    plt.title(f"Sample {sample_count}: {sample_id}", fontsize=14)

                    # Conversation display (right side)
                    plt.subplot(1, 2, 2)
                    plt.axis("off")

                    # Organize conversation text with proper line breaks
                    conv_text = f"ID: {sample_id}\n\n"
                    for j, conv in enumerate(conversations):
                        role = conv.get("from", "unknown")
                        content = conv.get("value", "").replace("<image>", "[image]")

                        # Wrap long text with proper line breaks
                        if len(content) > 300:
                            content = content[:300] + "..."

                        # Wrap text to fit display area
                        wrapped_content = textwrap.fill(content, width=40)
                        conv_text += f"{role.upper()}:\n{wrapped_content}\n\n"

                    plt.text(
                        0.05,
                        0.95,
                        conv_text,
                        transform=plt.gca().transAxes,
                        fontsize=9,
                        verticalalignment="top",
                        wrap=False,
                        bbox=dict(
                            boxstyle="round,pad=0.5",
                            facecolor="lightgray",
                            alpha=0.8,
                        ),
                        family="monospace",  # Use monospace font for better alignment
                    )

                    plt.tight_layout()
                    plt.show()

                    success_count += 1
                    print(f"✅ Display successful (total: {success_count})")

                    # Clear image from memory
                    del image

                else:
                    print(f"❌ Failed to get image: {image_filename}")
            else:
                print("❌ No image filename found")

            # Auto proceed to next sample
            if sample_count < max_samples:
                print(f"🔄 Streaming next sample... ({sample_count+1}/{max_samples})")

    except Exception as e:
        print(f"Streaming error: {e}")
        return

    print(f"\nCompleted: {success_count}/{sample_count} samples displayed successfully")
    print(f"Memory efficient: Only processed {sample_count} samples from large dataset")


if __name__ == "__main__":
    print("=== LLaVA Streaming Viewer ===\n")
    create_llava_streaming_viewer(max_samples=5)
