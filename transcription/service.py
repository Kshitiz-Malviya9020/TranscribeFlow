import os
import time
from config import Config
from models_loader import get_whisper_model, get_summarizer

processing_jobs = {}


# ============================================================
# SUMMARY FUNCTION
# ============================================================
def summarize_chunk(text_chunk):
    try:
        summ_model, summ_tokenizer = get_summarizer()

        input_ids = summ_tokenizer.encode(
            text_chunk,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        )

        summary_ids = summ_model.generate(
            input_ids,
            max_length=300,
            min_length=80,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=3
        )

        return summ_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    except Exception as e:
        print(f"Summary error: {e}")
        return text_chunk

# ============================================================
# FILE VALIDATION
# ============================================================
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def format_timestamp(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


# ============================================================
# MAIN TRANSCRIPTION FUNCTION
# ============================================================
def run_transcription(file_path, filename):

    print(f"Starting transcription for: {filename}")

    try:
        processing_jobs[filename] = {
            'status': 'processing',
            'progress': 5
        }

        if not os.path.exists(file_path):
            raise FileNotFoundError("Audio file not found")

        # 🔹 WHISPER TRANSCRIPTION
        processing_jobs[filename]['progress'] = 10

        model = get_whisper_model()
        result = model.transcribe(
            file_path,
            fp16=False,
            language="en"
        )


        processing_jobs[filename]['progress'] = 60

        full_text = result['text'].strip()
        segments = result.get('segments', [])

        formatted_transcript = ""

        for s in segments:
            start = format_timestamp(s['start'])
            end = format_timestamp(s['end'])
            formatted_transcript += f"[{start} - {end}] {s['text'].strip()}\n"

        processing_jobs[filename]['progress'] = 75

        # 🔹 SUMMARY (Single pass — stable & fast)
        summary_text = ""
        if len(full_text) > 100:
            summary_text = summarize_chunk(full_text)

        processing_jobs[filename]['progress'] = 85

        # ============================================================
        # 🔹 SAVE FILES (REQUIRED FOR EXPORT)
        # ============================================================

        transcript_path = os.path.join(
            Config.UPLOAD_FOLDER,
            filename + ".txt"
        )

        summary_path = os.path.join(
            Config.UPLOAD_FOLDER,
            filename + "_summary.txt"
        )

        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(formatted_transcript)

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_text)

        # ============================================================

        processing_jobs[filename].update({
            'status': 'completed',
            'progress': 100,
            'transcript': formatted_transcript,
            'summary': summary_text
        })

        print(f"Transcription completed for: {filename}")

    except Exception as e:

        print(f"ERROR in run_transcription: {e}")

        processing_jobs[filename] = {
            'status': 'error',
            'message': str(e),
            'progress': 0
        }
