import warnings
warnings.filterwarnings(
    "ignore",
    message=r"torch.utils.checkpoint: please pass in use_reentrant",
    category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message=r"The `vocab_size` attribute is deprecated.*",
    category=FutureWarning,
    module=r"transformers\.models\.llava\.configuration_llava",
)

from llava.train.train import train

if __name__ == "__main__":
    train()
