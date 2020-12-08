import os
from multiprocessing import cpu_count

CPU_NUM = cpu_count()

ROOT_PROJECT = os.path.abspath(__file__)[:-len("global_config.py")]

ROOT_UTILS = ROOT_PROJECT + "utils/"
ROOT_DATA = ROOT_PROJECT + "data/"
ROOT_SAVED_DATA = ROOT_PROJECT + "saved_data/"
ROOT_SAVED_MODEL = ROOT_PROJECT + "saved_model/"
ROOT_RESULT = ROOT_PROJECT + "result/"

TEST_MODE = False
BIG_GPU = False
CONVERT_DATA = False
POSTPROCESS = True
if BIG_GPU:
    BERT_MODEL = 'hfl/chinese-roberta-wwm-ext-large'
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
else:
    BERT_MODEL = 'hfl/chinese-roberta-wwm-ext'
    # BERT_MODEL = 'hfl/chinese-electra-base-discriminator'
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# BERT_MODEL = 'hfl/chinese-electra-base-generator'

print("\nTEST_MODE:", TEST_MODE,
      "\nBIG_GPU:", BIG_GPU,
      "\nCONVERT_DATA:", CONVERT_DATA,
      '\nPOSTPROCESS:', POSTPROCESS,
      '\nCUDA_VISIBLE_DEVICES:', os.environ["CUDA_VISIBLE_DEVICES"],
      '\nBERT_MODEL:', BERT_MODEL
      )
