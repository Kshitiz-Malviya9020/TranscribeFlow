import whisper
from transformers import BartForConditionalGeneration, BartTokenizer

_whisper_model = None
_summ_model = None
_summ_tokenizer = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        print("Loading Whisper Model...")
        _whisper_model = whisper.load_model("base")
    return _whisper_model


def get_summarizer():
    global _summ_model, _summ_tokenizer
    if _summ_model is None or _summ_tokenizer is None:
        print("Loading BART Model...")
        _summ_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        _summ_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    return _summ_model, _summ_tokenizer
