from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from transformers import MarianMTModel, MarianTokenizer
from youtube_transcript_api import YouTubeTranscriptApi
import re
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import nltk
from typing import List
import logging
from IndicTransToolkit import IndicProcessor

def get_youtube_video_id(url):
    return re.search("watch\?v=(\S{11})", url).group(1)

def extract_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return " ".join([t['text'] for t in transcript])

def summarize_text(llm, text):
    template = """Write a concise single paragraph summary of the transcript of a YouTube video.
    Output only the summary text and nothing else. Do not add a preface or postscript. The transcipt is as follows:
    {text}
    """
    prompt = PromptTemplate.from_template(template)
    return llm.invoke(prompt.format(text=text)).content


class IndicHindiTranslator:
    def __init__(self):
        self.model_name = "ai4bharat/indictrans2-en-indic-1B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, trust_remote_code=True)
        self.ip = IndicProcessor(inference=True)
        self.src_lang = "eng_Latn"
        self.tgt_lang = "hin_Deva"
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model.to(self.device)

    def translate(self, text: str) -> str:
        # Prepare input
        batch = self.ip.preprocess_batch([text], src_lang=self.src_lang, tgt_lang=self.tgt_lang)
        inputs = self.tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device)
        
        # Generate translation with optimized parameters
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
        )
        # Decode the generated tokens into text
        with self.tokenizer.as_target_tokenizer():
            translated_tokens = self.tokenizer.batch_decode(
                generated_tokens.detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        # Postprocess the translations, including entity replacement
        translation = self.ip.postprocess_batch(translated_tokens, lang=self.tgt_lang)[0]
        return translation

    def __del__(self):
        del self.model
        del self.tokenizer
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

class GemmaHindiTranslator:
    def __init__(self):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it", use_fast=False)
        
        print("Loading models...")
        self._initialize_model()

    def _initialize_model(self):
        self.base_model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-2b-it",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        ).to(self.device)
        
        self.model = PeftModel.from_pretrained(
            self.base_model,
            "monsterapi/gemma-2-2b-hindi-translator",
            torch_dtype=torch.float32
        ).to(self.device)
        print("Models loaded successfully")

    def translate(self, text: str) -> str:
        """Translate text to Hindi."""
        prompt = f"<bos><start_of_turn>user\nTranslate to Hindi: {text}<end_of_turn>\n<start_of_turn>model\n"
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192  # Using full context window
        ).to(self.device)
        
        try:
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=8192,  # Increased for longer translations
                    do_sample=True,
                    temperature=0.3,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # Extract translation from between markers
            start_marker = "<start_of_turn>model\n"
            end_marker = "<end_of_turn>"
            start_idx = output_text.find(start_marker) + len(start_marker)
            end_idx = output_text.find(end_marker, start_idx)
            
            if start_idx != -1 and end_idx != -1:
                return output_text[start_idx:end_idx].strip()
            return ""
                
        except Exception as e:
            print(f"Error during translation: {e}")
            return ""

    def __del__(self):
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'base_model'):
            del self.base_model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

class LlamaHindiTranslator: 
    def __init__(self, llm):
        llm = llm

    def translate(self, text):
        template = """Translate the following English text to Hindi:
        {text}
        """
        prompt = PromptTemplate.from_template(template)
        return llm.invoke(prompt.format(text=text)).content


if __name__ == '__main__':
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    video_url = input("Enter the YouTube video link: ")
    video_id = get_youtube_video_id(video_url)
    transcript = extract_transcript(video_id)

    llm = ChatOllama(model="llama3.3", temperature=0)
    summary = summarize_text(llm, transcript)
    print("Summary in English: ", summary)

    gemmaTranslator = GemmaHindiTranslator()
    gemma_hindi_summary = gemmaTranslator.translate(summary)
    print("Summary in Hindi generated by Gemma: ", gemma_hindi_summary)

    indicHindiTranslator = IndicHindiTranslator()
    helsinki_hindi_summary = indicHindiTranslator.translate(summary)
    print("Summary in Hindi generated by Indic: ", helsinki_hindi_summary)

    #llama_hindi_translator = LlamaHindiTranslator(llm)
    #llama_hindi_summary = llama_hindi_translator.translate(summary)
    #print("Summary in Hindi generated by LLAMA: ", llama_hindi_summary)
