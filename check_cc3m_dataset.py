import matplotlib.pyplot as plt
import webdataset as wds


def main(dataset_urls):
    # Use webdataset streaming to load CC3M dataset
    print(f"Loading CC3M dataset with streaming from {len(dataset_urls)} shards...")

    # Create webdataset with streaming settings
    dataset = (
        wds.WebDataset(dataset_urls, shardshuffle=True)  # Shuffle shards
        .shuffle(1000)  # Shuffle samples with buffer size 1000
        .decode("pil")  # PIL image decoding
    )

    for i, sample in enumerate(dataset):

        # Extract data
        image = sample.get("jpg")
        caption = "No Caption"

        # Get caption from json metadata or txt field
        if "json" in sample:
            json_data = sample["json"]
            if isinstance(json_data, dict) and "caption" in json_data:
                caption = json_data["caption"]
        elif "txt" in sample:
            caption = sample["txt"]

        print(f"Caption: {caption}")
        # Display information
        if image:
            plt.figure(figsize=(8, 6))
            plt.imshow(image)
            plt.axis("off")
            plt.title(f"Sample {i+1}: {caption[:50]}...")
            plt.show()

        if i >= 5:
            break

    print("Finish.")


if __name__ == "__main__":
    # CC3M has 576 tar files from 0000-0575
    base_url = "https://huggingface.co/datasets/pixparse/cc3m-wds/resolve/main/"

    # Load from 5 shards for testing
    dataset_urls = [f"{base_url}cc3m-train-{i:04d}.tar" for i in range(5)]
    main(dataset_urls)
