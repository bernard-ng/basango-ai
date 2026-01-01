DATASETS_URL = [
    "https://huggingface.co/datasets/bernard-ng/drc-news-corpus/resolve/main/actualite.cd.csv?download=true",
    "https://huggingface.co/datasets/bernard-ng/drc-news-corpus/resolve/main/beto.cd.csv?download=true",
    "https://huggingface.co/datasets/bernard-ng/drc-news-corpus/resolve/main/mediacongo.net.csv?download=true",
    "https://huggingface.co/datasets/bernard-ng/drc-news-corpus/resolve/main/radiookapi.net.csv?download=true",
]

CONCURRENCY = 4
CHUNK_SIZE = 1024 * 1024

BATCH_SIZE = 64
TORCH_NUM_THREADS = 8
TORCH_NUM_INTEROP_THREADS = 2
