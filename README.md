
# Translation Pipeline for CSV Files

This Python script allows you to translate CSV files using Hugging Face's "Helsinki-NLP/opus-mt-en-fr" translation model. The script is designed to be used with GPUs, but it will fall back to CPU if no GPU is available.

## Prerequisites

- Python 3.7 or higher
- PyTorch 1.8.1 or higher
- Transformers 4.6.1 or higher
- Colorama 0.4.4 or higher
- Pandas 1.2.4 or higher

These dependencies can be installed with pip:

```bash
pip install torch transformers colorama pandas
```

## Usage

1. Prepare your source CSV files in a folder. Each file should be a CSV file, where each row is an entry to translate.

2. Run the Python script. For example:

```bash
python translate_tool.py
```

3. The script will first split the source files into several smaller files, to facilitate parallelizing the translation.

4. Then, the script will translate each file in parallel, using one process for each available GPU. If no GPU is available, the script will use the CPU.

5. After translation, the script will merge all translated files into a single CSV file.

## Script Parameters

You can modify the following parameters at the top of the script:

- `src_dir`: The path to the folder containing the source CSV files to translate.
- `split_dir`: The path to the folder where the source files will be split.
- `split_trad_dir`: The path to the folder where the translated files will be saved.
- `merge_file`: The path to the file where the translated files will be merged.
- `batch_size`: The batch size for translation. Increase this for faster translation if you have enough GPU memory.

## Notes

- This script translates text from English to French. If you want to translate into a different language, you can change the model name accordingly.
- This script uses `torch.multiprocessing` for parallelization. Ensure that your system supports multiprocessing with PyTorch and CUDA.
- This script uses `colorama` for coloring printed messages. If you do not want colors, you can remove the references to `Fore`.

## Acknowledgements

This script uses the "Helsinki-NLP/opus-mt-en-fr" model from Hugging Face's model hub. We would like to thank [Helsinki-NLP](https://huggingface.co/Helsinki-NLP) for providing this model.

## License

This script is licensed under the MIT License. See the LICENSE file for details.
